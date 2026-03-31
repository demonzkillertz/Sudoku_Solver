/// Sudoku solver using backtracking algorithm - matches main.py
class SudokuSolver {
  /// Solves the board in-place, returns true if solved
  static bool solveBoard(List<List<int>> board) {
    for (int row = 0; row < 9; row++) {
      for (int col = 0; col < 9; col++) {
        if (board[row][col] == 0) {
          for (int num = 1; num <= 9; num++) {
            if (_isValid(board, row, col, num)) {
              board[row][col] = num;
              if (solveBoard(board)) return true;
              board[row][col] = 0;
            }
          }
          return false;
        }
      }
    }
    return true;
  }

  static bool _isValid(List<List<int>> board, int row, int col, int num) {
    // Check row and column
    for (int x = 0; x < 9; x++) {
      if (board[row][x] == num || board[x][col] == num) return false;
    }
    // Check 3x3 box
    final boxRow = (row ~/ 3) * 3;
    final boxCol = (col ~/ 3) * 3;
    for (int r = boxRow; r < boxRow + 3; r++) {
      for (int c = boxCol; c < boxCol + 3; c++) {
        if (board[r][c] == num) return false;
      }
    }
    return true;
  }

  /// Validate a board for basic Sudoku rules
  static Map<String, dynamic> validateBoard(List<List<int>> board) {
    // Count filled cells
    int filledCount = 0;
    for (var row in board) {
      for (var cell in row) {
        if (cell != 0) filledCount++;
      }
    }

    // Check if board is too empty (likely failed detection)
    if (filledCount < 10) {
      return {
        'valid': false,
        'error': 'Too few digits detected ($filledCount). Try a clearer image.'
      };
    }

    // Check rows for duplicates
    for (int r = 0; r < 9; r++) {
      final seen = <int>{};
      for (int c = 0; c < 9; c++) {
        final val = board[r][c];
        if (val != 0) {
          if (seen.contains(val)) {
            return {'valid': false, 'error': 'Duplicate $val in row ${r + 1}'};
          }
          seen.add(val);
        }
      }
    }

    // Check columns for duplicates
    for (int c = 0; c < 9; c++) {
      final seen = <int>{};
      for (int r = 0; r < 9; r++) {
        final val = board[r][c];
        if (val != 0) {
          if (seen.contains(val)) {
            return {
              'valid': false,
              'error': 'Duplicate $val in column ${c + 1}'
            };
          }
          seen.add(val);
        }
      }
    }

    // Check 3x3 boxes for duplicates
    for (int boxRow = 0; boxRow < 3; boxRow++) {
      for (int boxCol = 0; boxCol < 3; boxCol++) {
        final seen = <int>{};
        for (int r = boxRow * 3; r < boxRow * 3 + 3; r++) {
          for (int c = boxCol * 3; c < boxCol * 3 + 3; c++) {
            final val = board[r][c];
            if (val != 0) {
              if (seen.contains(val)) {
                return {
                  'valid': false,
                  'error':
                      'Duplicate $val in box (${boxRow + 1},${boxCol + 1})'
                };
              }
              seen.add(val);
            }
          }
        }
      }
    }

    return {'valid': true, 'error': null};
  }
}
