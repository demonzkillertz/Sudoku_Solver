/// Sudoku board model
class SudokuBoard {
  List<List<int>> cells;

  SudokuBoard(this.cells);

  factory SudokuBoard.empty() =>
      SudokuBoard(List.generate(9, (_) => List.filled(9, 0)));

  void clear() {
    for (var row in cells) {
      for (int i = 0; i < 9; i++) {
        row[i] = 0;
      }
    }
  }

  List<List<int>> copy() {
    return List.generate(9, (i) => List<int>.from(cells[i]));
  }

  int get filledCount {
    int count = 0;
    for (var row in cells) {
      for (var cell in row) {
        if (cell != 0) count++;
      }
    }
    return count;
  }

  bool get isEmpty => filledCount == 0;
}
