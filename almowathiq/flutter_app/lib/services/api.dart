import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

class ApiService {
  static const bool useMock = false;

  static const String mockStatus = 'غير موثّق';

  static Future<String> verifyImage(Uint8List imageBytes) async {
    if (useMock) {
      return _mockResponse();
    }

    try {
      final uri = Uri.parse('https://almowathiq.onrender.com/verify');

      final request = http.MultipartRequest('POST', uri);

      request.files.add(
        http.MultipartFile.fromBytes(
          'image',
          imageBytes,
          filename: 'test_image.jpg',
          contentType: MediaType('image', 'jpeg'),
        ),
      );

      print('API: إرسال الصورة إلى $uri');
      print('API: حجم الصورة = ${imageBytes.length} bytes');

      final response = await request.send();

      final responseBody = await response.stream.bytesToString();

      print('API: status = ${response.statusCode}');
      print('API: response = $responseBody');

      if (response.statusCode == 200) {
        return responseBody;
      }

      throw Exception('الخادم أعاد حالة ${response.statusCode}\n$responseBody');
    } catch (e) {
      print('API ERROR: $e');
      rethrow;
    }
  }

  static String _mockResponse() {
    switch (mockStatus) {
      case 'موثّق':
        return jsonEncode({
          'content_type': 'fatwa',
          'status': 'موثّق',
          'confidence': 0.98,
          'extracted_text': 'هذا نص تجريبي مستخرج من الصورة.',
          'correct_text': 'هذا نص تجريبي يمثل النص الكامل من المصدر.',
          'missing_context': [],
          'explanation':
              'تم العثور على تطابق موثوق بين النص الظاهر في الصورة والمصدر الموجود في قاعدة المعرفة.',
          'source': {
            'scholar': 'مصدر تجريبي',
            'title': 'مصدر تجريبي للتحقق',
            'url': 'https://example.com',
          },
        });

      case 'محرّف أو مقتطع':
        return jsonEncode({
          'content_type': 'fatwa',
          'status': 'محرّف أو مقتطع',
          'confidence': 0.91,
          'extracted_text': 'يجوز فعل هذا الأمر في جميع الحالات.',
          'correct_text':
              'يجوز فعل هذا الأمر عند تحقق شروط معينة وفي حالات محددة.',
          'missing_context': [
            'تم حذف جزء من النص الأصلي.',
            'السياق الكامل يوضح وجود شروط للحكم.',
          ],
          'explanation':
              'تم العثور على أصل للنص، لكن النص الظاهر في الصورة يختلف عن المصدر أو يفتقد جزءًا من سياقه.',
          'source': {
            'scholar': 'مصدر تجريبي',
            'title': 'مصدر تجريبي للتحقق',
            'url': 'https://example.com',
          },
        });

      case 'غير موثّق':
        return jsonEncode({
          'content_type': 'fatwa',
          'status': 'غير موثّق',
          'confidence': 0.35,
          'extracted_text': 'هذا نص تجريبي غير موجود في المصادر المستخدمة.',
          'correct_text': null,
          'missing_context': [],
          'explanation':
              'لم نجد هذا النص في مصادرنا الموثوقة. وهذا لا يعني بالضرورة أنه خاطئ، وإنما لم نتمكن من توثيقه ضمن قاعدة المعرفة المستخدمة.',
          'source': null,
        });

      default:
        return jsonEncode({
          'content_type': 'fatwa',
          'status': 'غير موثّق',
          'confidence': 0.0,
          'extracted_text': '',
          'correct_text': null,
          'missing_context': [],
          'explanation': 'تعذر تحديد حالة التحقق.',
          'source': null,
        });
    }
  }
}
