import 'package:opencv_dart/opencv_dart.dart' as cv; void main() { var mat = cv.Mat.empty(); cv.resize(mat, (10, 10), interpolation: cv.INTER_AREA); }
