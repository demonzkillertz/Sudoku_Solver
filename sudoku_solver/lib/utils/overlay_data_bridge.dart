/// Shared data bridge for overlay communication
class OverlayDataBridge {
  static List<List<int>> solvedBoard =
      List.generate(9, (_) => List.filled(9, 0));
  static List<List<int>> originalBoard =
      List.generate(9, (_) => List.filled(9, 0));
  static double gridLeft = 70;
  static double gridTop = 420;
  static double cellSize = 120;
  static String status = 'Ready';

  static Map<String, dynamic> toJson() {
    return {
      'solved': solvedBoard,
      'original': originalBoard,
      'gridLeft': gridLeft,
      'gridTop': gridTop,
      'cellSize': cellSize,
      'status': status,
    };
  }

  static void fromJson(Map<String, dynamic> json) {
    if (json['solved'] != null) {
      solvedBoard = (json['solved'] as List)
          .map((row) => (row as List).map((e) => e as int).toList())
          .toList();
    }
    if (json['original'] != null) {
      originalBoard = (json['original'] as List)
          .map((row) => (row as List).map((e) => e as int).toList())
          .toList();
    }
    gridLeft = (json['gridLeft'] ?? 70).toDouble();
    gridTop = (json['gridTop'] ?? 420).toDouble();
    cellSize = (json['cellSize'] ?? 120).toDouble();
    status = json['status'] ?? 'Ready';
  }

  static void clear() {
    solvedBoard = List.generate(9, (_) => List.filled(9, 0));
    originalBoard = List.generate(9, (_) => List.filled(9, 0));
    status = 'Ready';
  }
}
