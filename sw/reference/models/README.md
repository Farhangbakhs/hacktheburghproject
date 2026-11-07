# Face tracking models

Used by `../face_tracker.py`. Fetch or verify them with:

```
.venv/Scripts/python models/download_models.py
```

The script is safe to re-run: files that already match the expected size and hash are skipped.

| File | Size | Source | Licence | Used for |
|---|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | 232,589 B, sha256 `8f2383e4...52fa4` | [opencv_zoo/models/face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (Git LFS) | MIT | `cv2.FaceDetectorYN` face box, 5 keypoints, score |
| `lbfmodel.yaml` | 56,375,857 B, git blob sha1 `417d6db3...b1610` | [kurnianggoro/GSOC2017/data](https://github.com/kurnianggoro/GSOC2017/tree/master/data), the model the OpenCV facemark tutorial links to | from the OpenCV GSoC 2017 facemark project, trained on 300-W (iBUG) | `cv2.face.FacemarkLBF` 68 landmarks |

`lbfmodel.yaml` is git-ignored because it is larger than 5 MB. The ONNX file is small and can be committed.

If files are missing, `FaceTracker` falls back and reports it in `tracker.backend`:
`yunet+lbf` (both files) -> `yunet` (ONNX only) -> `haar` (neither file, uses OpenCV's bundled cascade).

Note: the 300-W training data is licensed for research use. Check that this fits your use before shipping the LBF model in a product. A hackathon demo is fine.
