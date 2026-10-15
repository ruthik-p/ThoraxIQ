"""Grayscale preprocessing and HOG/LBP/GLCM feature extraction."""
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

IMAGE_SIZE = (64, 64)
FEATURE_VERSION = "hog-lbp-glcm-v1-64"


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Apply the shared grayscale conversion and resize used in train and app."""
    with image:
        image = image.convert("L").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def read_gray(image_name: str, image_source: str | Path, archive_handle=None) -> np.ndarray:
    """Read one image from a directory or images.zip without extracting the archive."""
    source = Path(image_source)
    try:
        if source.is_dir():
            path = source / image_name
            if not path.is_file():
                raise FileNotFoundError(path)
            image = Image.open(path)
        elif source.suffix.lower() == ".zip":
            def load_from_zip(archive):
                import io
                try:
                    member = archive.getinfo(f"images/{image_name}")
                except KeyError:
                    try:
                        member = archive.getinfo(image_name)
                    except KeyError as exc:
                        raise FileNotFoundError(f"{image_name} not found in {source}") from exc
                return Image.open(io.BytesIO(archive.read(member)))
            if archive_handle is None:
                with zipfile.ZipFile(source) as archive:
                    image = load_from_zip(archive)
            else:
                image = load_from_zip(archive_handle)
        else:
            raise ValueError(f"Image source must be a directory or zip file: {source}")
        return preprocess_image(image)
    except (UnidentifiedImageError, OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Could not read image {image_name}: {exc}") from exc


def extract_features(gray: np.ndarray) -> np.ndarray:
    """Build a fixed-length feature vector from a 64x64 grayscale image."""
    if gray.shape != (64, 64):
        raise ValueError(f"Expected a 64x64 image, got {gray.shape}")
    unit = gray.astype(np.float32) / 255.0
    hog_features = hog(unit, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10), density=True)
    quantized = (gray // 16).astype(np.uint8)
    matrix = graycomatrix(quantized, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                          levels=16, symmetric=True, normed=True)
    glcm = np.concatenate([graycoprops(matrix, prop).ravel()
                           for prop in ("contrast", "homogeneity", "energy", "correlation")])
    return np.concatenate([hog_features, lbp_hist, glcm]).astype(np.float32)


def extract_dataset(frame, image_source: str | Path, cache_path: str | Path,
                    force: bool = False):
    """Extract features row-by-row, recording unreadable images and caching results."""
    import hashlib
    cache_path = Path(cache_path)
    image_names = frame["Image Index"].astype(str).tolist()
    source = Path(image_source)
    signature = [str(source.resolve())]
    if source.is_dir():
        for name in image_names:
            try:
                stat = (source / name).stat()
                signature.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
            except OSError:
                signature.append(f"{name}:missing")
    else:
        try:
            stat = source.stat()
            signature.append(f"{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            signature.append("missing")
    key_source = "\n".join(image_names).encode()
    source_signature = "\n".join(signature).encode()
    key = hashlib.sha256(key_source + FEATURE_VERSION.encode() + source_signature).hexdigest()
    if cache_path.exists() and not force:
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["key"].item()) == key:
            return cached["features"], cached["valid_indices"].astype(int), cached["errors"].astype(str)
    vectors, valid, errors = [], [], []
    archive = zipfile.ZipFile(image_source) if Path(image_source).suffix.lower() == ".zip" else None
    try:
        for idx, name in enumerate(image_names):
            try:
                vectors.append(extract_features(read_gray(name, image_source, archive)))
                valid.append(idx)
            except (FileNotFoundError, ValueError, OSError) as exc:
                errors.append(f"{name}: {exc}")
    finally:
        if archive is not None:
            archive.close()
    if not vectors:
        raise ValueError("No readable images found. Check --images path and CSV filenames.")
    features = np.stack(vectors)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, key=np.asarray(key), features=features,
                        valid_indices=np.asarray(valid, dtype=np.int64), errors=np.asarray(errors, dtype=str))
    return features, np.asarray(valid, dtype=np.int64), np.asarray(errors, dtype=str)

