import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:url_launcher/url_launcher.dart';

import 'services/api.dart';

void main() {
  runApp(const AlMowathiqApp());
}

class AlMowathiqApp extends StatelessWidget {
  const AlMowathiqApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'الموثّق الذكي',
      theme: ThemeData(
        useMaterial3: true,
        fontFamily: 'Arial',
        scaffoldBackgroundColor: const Color(0xFFF8F5ED),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF123C32),
          brightness: Brightness.light,
        ),
      ),
      home: const HomePage(),
    );
  }
}

// ============================================================
// الألوان
// ============================================================

const Color deepGreen = Color(0xFF123C32);
const Color green = Color(0xFF1E5A4A);
const Color lightGreen = Color(0xFFE8F0EB);
const Color ivory = Color(0xFFF8F5ED);
const Color white = Color(0xFFFFFFFF);
const Color gold = Color(0xFFD6B56C);
const Color lightGold = Color(0xFFE9D7A8);
const Color darkText = Color(0xFF20312C);
const Color mutedText = Color(0xFF68736E);

// ============================================================
// زخرفة إسلامية هندسية
// ============================================================

class IslamicPattern extends StatelessWidget {
  final double opacity;
  final bool goldOnly;

  const IslamicPattern({super.key, this.opacity = 0.20, this.goldOnly = false});

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: CustomPaint(
        painter: IslamicPatternPainter(opacity: opacity, goldOnly: goldOnly),
      ),
    );
  }
}

class IslamicPatternPainter extends CustomPainter {
  final double opacity;
  final bool goldOnly;

  IslamicPatternPainter({required this.opacity, required this.goldOnly});

  @override
  void paint(Canvas canvas, Size size) {
    const double spacing = 58;

    final Paint linePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1
      ..color = (goldOnly ? gold : deepGreen).withOpacity(opacity);

    final Paint secondPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 0.7
      ..color = gold.withOpacity(opacity * 0.75);

    for (double x = -spacing; x < size.width + spacing; x += spacing) {
      for (double y = -spacing; y < size.height + spacing; y += spacing) {
        _drawIslamicStar(canvas, Offset(x, y), 20, linePaint, secondPaint);
      }
    }
  }

  void _drawIslamicStar(
    Canvas canvas,
    Offset center,
    double radius,
    Paint outerPaint,
    Paint innerPaint,
  ) {
    final Path outer = Path();
    final Path inner = Path();

    for (int i = 0; i < 8; i++) {
      final double angle = (-math.pi / 2) + (i * math.pi / 4);

      final Offset p = Offset(
        center.dx + math.cos(angle) * radius,
        center.dy + math.sin(angle) * radius,
      );

      if (i == 0) {
        outer.moveTo(p.dx, p.dy);
      } else {
        outer.lineTo(p.dx, p.dy);
      }
    }

    outer.close();

    for (int i = 0; i < 4; i++) {
      final double angle = (-math.pi / 4) + (i * math.pi / 2);

      final Offset p = Offset(
        center.dx + math.cos(angle) * radius * 0.67,
        center.dy + math.sin(angle) * radius * 0.67,
      );

      if (i == 0) {
        inner.moveTo(p.dx, p.dy);
      } else {
        inner.lineTo(p.dx, p.dy);
      }
    }

    inner.close();

    canvas.drawPath(outer, outerPaint);
    canvas.drawPath(inner, innerPaint);

    canvas.drawLine(
      Offset(center.dx - radius, center.dy),
      Offset(center.dx + radius, center.dy),
      innerPaint,
    );

    canvas.drawLine(
      Offset(center.dx, center.dy - radius),
      Offset(center.dx, center.dy + radius),
      innerPaint,
    );
  }

  @override
  bool shouldRepaint(covariant IslamicPatternPainter oldDelegate) {
    return oldDelegate.opacity != opacity || oldDelegate.goldOnly != goldOnly;
  }
}

// ============================================================
// HOME
// ============================================================

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ImagePicker _picker = ImagePicker();

  Uint8List? _selectedImage;
  bool _isVerifying = false;

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? picked = await _picker.pickImage(
        source: source,
        imageQuality: 90,
      );

      if (picked == null) return;

      final Uint8List bytes = await picked.readAsBytes();

      setState(() {
        _selectedImage = bytes;
      });
    } catch (e) {
      _showMessage('تعذر اختيار الصورة، حاول مرة أخرى');
    }
  }

  void _removeImage() {
    setState(() {
      _selectedImage = null;
    });
  }

  Future<void> _verifyImage() async {
    if (_selectedImage == null) {
      _showMessage('اختر صورة أولًا');
      return;
    }

    setState(() {
      _isVerifying = true;
    });

    try {
      final String response = await ApiService.verifyImage(_selectedImage!);

      final dynamic result = jsonDecode(response);

      if (!mounted) return;

      Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => ResultPage(result: result)),
      );
    } catch (e) {
      if (!mounted) return;

      _showMessage(
        'تعذر الاتصال بخدمة التحقق\n'
        'تأكد من تشغيل الخادم ثم حاول مرة أخرى',
      );
    } finally {
      if (mounted) {
        setState(() {
          _isVerifying = false;
        });
      }
    }
  }

  void _showMessage(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          message,
          textAlign: TextAlign.right,
          style: const TextStyle(fontSize: 14),
        ),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Directionality(
      textDirection: TextDirection.rtl,
      child: Scaffold(
        body: SafeArea(
          child: Stack(
            children: [
              Positioned(
                top: -40,
                right: -30,
                child: SizedBox(
                  width: 230,
                  height: 180,
                  child: Opacity(
                    opacity: 0.45,
                    child: IslamicPattern(opacity: 0.18, goldOnly: true),
                  ),
                ),
              ),
              Positioned(
                bottom: -30,
                left: -20,
                child: SizedBox(
                  width: 210,
                  height: 170,
                  child: Opacity(
                    opacity: 0.35,
                    child: IslamicPattern(opacity: 0.16, goldOnly: false),
                  ),
                ),
              ),
              SingleChildScrollView(
                padding: const EdgeInsets.symmetric(
                  horizontal: 22,
                  vertical: 18,
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    _buildTopBar(),
                    const SizedBox(height: 26),
                    _buildHero(),
                    const SizedBox(height: 22),
                    _buildTrustBanner(),
                    const SizedBox(height: 24),
                    _buildActions(),
                    if (_selectedImage != null) ...[
                      const SizedBox(height: 20),
                      _buildImagePreview(),
                    ],
                    const SizedBox(height: 30),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTopBar() {
    return Row(
      children: [
        Container(
          width: 52,
          height: 52,
          decoration: BoxDecoration(
            color: deepGreen,
            borderRadius: BorderRadius.circular(16),
          ),
          child: Stack(
            children: [
              Positioned.fill(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(16),
                  child: IslamicPattern(opacity: 0.14, goldOnly: true),
                ),
              ),
              const Center(
                child: Icon(
                  Icons.verified_outlined,
                  color: lightGold,
                  size: 27,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(width: 13),
        const Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'الموثّق الذكي',
              style: TextStyle(
                color: deepGreen,
                fontSize: 21,
                fontWeight: FontWeight.w800,
              ),
            ),
            SizedBox(height: 2),
            Text(
              'تحقق قبل أن تنشر',
              style: TextStyle(color: mutedText, fontSize: 12),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildHero() {
    return Container(
      height: 285,
      decoration: BoxDecoration(
        color: deepGreen,
        borderRadius: BorderRadius.circular(30),
      ),
      clipBehavior: Clip.antiAlias,
      child: Stack(
        children: [
          Positioned(
            left: -35,
            top: -35,
            child: SizedBox(
              width: 260,
              height: 230,
              child: IslamicPattern(opacity: 0.17, goldOnly: true),
            ),
          ),
          Positioned(
            right: -45,
            bottom: -40,
            child: SizedBox(
              width: 270,
              height: 210,
              child: IslamicPattern(opacity: 0.11, goldOnly: false),
            ),
          ),
          Positioned(
            top: 28,
            left: 25,
            child: CustomPaint(
              size: const Size(70, 70),
              painter: LargeIslamicStarPainter(),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(28),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 52,
                  height: 52,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.10),
                    borderRadius: BorderRadius.circular(15),
                    border: Border.all(color: gold.withOpacity(0.45)),
                  ),
                  child: const Icon(
                    Icons.menu_book_rounded,
                    color: lightGold,
                    size: 26,
                  ),
                ),
                const Spacer(),
                const Text(
                  'هل هذا الحديث أو الفتوى',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 28,
                    fontWeight: FontWeight.w800,
                    height: 1.25,
                  ),
                ),
                const Text(
                  'منسوب إلى مصدره فعلًا؟',
                  style: TextStyle(
                    color: lightGold,
                    fontSize: 28,
                    fontWeight: FontWeight.w800,
                    height: 1.25,
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  'ارفع صورة الحديث أو الفتوى، ودع الموثّق الذكي يتحقق منها من مصادر موثوقة.',
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.75),
                    fontSize: 14,
                    height: 1.7,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTrustBanner() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFF0E8D5),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: gold.withOpacity(0.30)),
      ),
      child: Row(
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              color: white,
              borderRadius: BorderRadius.circular(13),
            ),
            child: const Icon(
              Icons.shield_outlined,
              color: deepGreen,
              size: 23,
            ),
          ),
          const SizedBox(width: 12),
          const Expanded(
            child: Text(
              'التحقق من المعلومة قبل نشرها يساعد على الحد من تداول الأحاديث والفتاوى غير الموثوقة.',
              style: TextStyle(
                color: darkText,
                fontSize: 13,
                height: 1.6,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActions() {
    return Column(
      children: [
        SizedBox(
          height: 58,
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: _isVerifying
                ? null
                : () => _pickImage(ImageSource.gallery),
            icon: const Icon(Icons.image_outlined, size: 22),
            label: const Text(
              'رفع صورة حديث أو فتوى',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
            ),
            style: ElevatedButton.styleFrom(
              backgroundColor: deepGreen,
              foregroundColor: white,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(17),
              ),
              elevation: 0,
            ),
          ),
        ),
        const SizedBox(height: 12),
        SizedBox(
          height: 56,
          width: double.infinity,
          child: OutlinedButton.icon(
            onPressed: _isVerifying
                ? null
                : () => _pickImage(ImageSource.camera),
            icon: const Icon(Icons.camera_alt_outlined, size: 21),
            label: const Text(
              'التقاط صورة',
              style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
            ),
            style: OutlinedButton.styleFrom(
              foregroundColor: deepGreen,
              side: BorderSide(color: deepGreen.withOpacity(0.35)),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(17),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildImagePreview() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: const Color(0xFFE4DED0)),
      ),
      child: Column(
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: Image.memory(
              _selectedImage!,
              width: double.infinity,
              height: 280,
              fit: BoxFit.cover,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: OutlinedButton(
                  onPressed: _isVerifying ? null : _removeImage,
                  child: const Text('اختيار صورة أخرى'),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: ElevatedButton(
                  onPressed: _isVerifying ? null : _verifyImage,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: deepGreen,
                    foregroundColor: white,
                  ),
                  child: _isVerifying
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: white,
                          ),
                        )
                      : const Text('ابدأ التحقق'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ============================================================
// نجمة إسلامية كبيرة
// ============================================================

class LargeIslamicStarPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final Offset center = Offset(size.width / 2, size.height / 2);

    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.1
      ..color = gold.withOpacity(0.55);

    final Paint innerPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 0.8
      ..color = lightGold.withOpacity(0.35);

    final double radius = size.width * 0.43;

    final Path star = Path();

    for (int i = 0; i < 8; i++) {
      final double angle = -math.pi / 2 + i * math.pi / 4;

      final Offset point = Offset(
        center.dx + math.cos(angle) * radius,
        center.dy + math.sin(angle) * radius,
      );

      if (i == 0) {
        star.moveTo(point.dx, point.dy);
      } else {
        star.lineTo(point.dx, point.dy);
      }
    }

    star.close();

    canvas.drawPath(star, paint);

    final Path square1 = Path()
      ..moveTo(center.dx - radius * 0.62, center.dy)
      ..lineTo(center.dx, center.dy - radius * 0.62)
      ..lineTo(center.dx + radius * 0.62, center.dy)
      ..lineTo(center.dx, center.dy + radius * 0.62)
      ..close();

    final Path square2 = Path()
      ..moveTo(center.dx - radius * 0.62, center.dy)
      ..lineTo(center.dx, center.dy - radius * 0.62)
      ..lineTo(center.dx + radius * 0.62, center.dy)
      ..lineTo(center.dx, center.dy + radius * 0.62)
      ..close();

    canvas.drawPath(square1, innerPaint);
    canvas.drawPath(square2, innerPaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return false;
  }
}

// ============================================================
// RESULT PAGE
// ============================================================

class ResultPage extends StatelessWidget {
  final dynamic result;

  const ResultPage({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final String status = result['status']?.toString() ?? '';

    final String displayStatus = status == 'موثّق'
        ? 'موثّق'
        : status == 'يحتاج سياق' || status == 'يحتاج تصحيح'
        ? 'يحتاج تصحيح'
        : 'غير موثّق';

    final String extractedText = result['extracted_text']?.toString() ?? '';

    final String correctText = result['correct_text']?.toString() ?? '';

    final String explanation = result['explanation']?.toString() ?? '';

    final String evidence = result['evidence']?.toString() ?? '';

    // سبب "يحتاج تصحيح": truncated / altered / not_authentic
    final String issue = result['issue']?.toString() ?? '';

    final List<String> issues = result['issues'] is List
        ? (result['issues'] as List).map((e) => e.toString()).toList()
        : <String>[];

    final String correctTextTitle = issue == 'not_authentic'
        ? 'البديل الصحيح'
        : 'النص كما ورد في المصدر';

    List<String> _words(dynamic list) => list is List
        ? list
              .expand((e) => e.toString().split(RegExp(r'\s+')))
              .map(_cleanWord)
              .where((w) => w.isNotEmpty)
              .toList()
        : <String>[];

    // الكلمات المضافة أو المغيّرة في الصورة (أحمر) والمحذوفة من المصدر (أخضر)
    final Set<String> addedWords = _words(result['added_words']).toSet();
    final Set<String> removedWords = _words(result['removed_words']).toSet();

    // الجملة المقابلة من المصدر (بدل النص كاملًا)
    final String sourceExcerpt = result['source_excerpt']?.toString() ?? '';

    final dynamic source = result['source'];

    final dynamic alternativeSource = result['alternative_source'];

    return Directionality(
      textDirection: TextDirection.rtl,
      child: Scaffold(
        backgroundColor: ivory,
        appBar: AppBar(
          backgroundColor: ivory,
          elevation: 0,
          centerTitle: true,
          title: const Text(
            'نتيجة التحقق',
            style: TextStyle(color: deepGreen, fontWeight: FontWeight.w800),
          ),
          iconTheme: const IconThemeData(color: deepGreen),
        ),
        body: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(20, 8, 20, 30),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildResultHeader(displayStatus),
              const SizedBox(height: 18),
              _buildStatusCard(displayStatus, issue, issues),
              if (extractedText.isNotEmpty) ...[
                const SizedBox(height: 18),
                _buildTextSection(
                  title: 'النص المستخرج',
                  text: extractedText,
                  icon: Icons.text_snippet_outlined,
                  highlight: addedWords,
                  highlightColor: const Color(0xFFC0392B),
                ),
              ],
              if (correctText.isNotEmpty && sourceExcerpt.isNotEmpty) ...[
                const SizedBox(height: 18),
                _buildTextSection(
                  title: 'النص الصحيح من المصدر',
                  text: sourceExcerpt,
                  icon: Icons.fact_check_outlined,
                  highlight: removedWords,
                  highlightColor: const Color(0xFF1E7B4F),
                  fullText: correctText,
                ),
              ] else if (correctText.isNotEmpty) ...[
                const SizedBox(height: 18),
                _buildTextSection(
                  title: correctTextTitle,
                  text: correctText,
                  icon: Icons.fact_check_outlined,
                  highlight: removedWords,
                  highlightColor: const Color(0xFF1E7B4F),
                ),
              ],
              if (evidence.isNotEmpty) ...[
                const SizedBox(height: 18),
                _buildTextSection(
                  title: 'الأدلة',
                  text: evidence,
                  icon: Icons.menu_book_outlined,
                ),
              ],
              if (explanation.isNotEmpty) ...[
                const SizedBox(height: 18),
                _buildTextSection(
                  title: 'التوضيح',
                  text: explanation,
                  icon: Icons.info_outline,
                ),
              ],
              if (source is Map && source['scholar'] != null) ...[
                const SizedBox(height: 18),
                _buildSourceCard(
                  context,
                  source,
                  heading: issue == 'not_authentic' ? 'حكم الحديث' : 'المصدر',
                ),
              ],
              if (alternativeSource is Map) ...[
                const SizedBox(height: 18),
                _buildSourceCard(
                  context,
                  alternativeSource,
                  heading: 'مصدر البديل الصحيح',
                ),
              ],
              const SizedBox(height: 24),
              SizedBox(
                height: 56,
                child: ElevatedButton.icon(
                  onPressed: () {
                    Navigator.pop(context);
                  },
                  icon: const Icon(Icons.refresh_rounded),
                  label: const Text(
                    'التحقق من صورة أخرى',
                    style: TextStyle(fontWeight: FontWeight.w700),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: deepGreen,
                    foregroundColor: white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(17),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildResultHeader(String displayStatus) {
    final bool verified = displayStatus == 'موثّق';

    final bool altered = displayStatus == 'يحتاج تصحيح';

    IconData icon;

    if (verified) {
      icon = Icons.verified_rounded;
    } else if (altered) {
      icon = Icons.info_rounded;
    } else {
      icon = Icons.search_off_rounded;
    }

    return Container(
      height: 180,
      decoration: BoxDecoration(
        color: deepGreen,
        borderRadius: BorderRadius.circular(28),
      ),
      clipBehavior: Clip.antiAlias,
      child: Stack(
        children: [
          Positioned(
            left: -35,
            top: -25,
            child: SizedBox(
              width: 230,
              height: 200,
              child: IslamicPattern(opacity: 0.16, goldOnly: true),
            ),
          ),
          Positioned(
            right: -35,
            bottom: -30,
            child: SizedBox(
              width: 220,
              height: 190,
              child: IslamicPattern(opacity: 0.10),
            ),
          ),
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Container(
                  width: 62,
                  height: 62,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.10),
                    shape: BoxShape.circle,
                    border: Border.all(color: gold.withOpacity(0.45)),
                  ),
                  child: Icon(icon, color: lightGold, size: 34),
                ),
                const SizedBox(height: 12),
                const Text(
                  'نتيجة التحقق',
                  style: TextStyle(color: Colors.white70, fontSize: 13),
                ),
                const SizedBox(height: 4),
                Text(
                  displayStatus.isEmpty ? 'غير معروف' : displayStatus,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 25,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusCard(String status, String issue, List<String> issues) {
    late String title;
    late String description;
    late IconData icon;
    late Color background;

    if (status == 'موثّق') {
      if (issue == 'paraphrased') {
        title = 'موثّق: منقول بالمعنى';
        description =
            'النص في الصورة منقول بالمعنى، والحكم مطابق للمصدر. وهذا لفظ المصدر كما ورد.';
      } else if (issue == 'abridged') {
        title = 'موثّق: النص مختصر من المصدر';
        description =
            'النص الظاهر في الصورة جزء من المصدر، والجزء المحذوف لا يغيّر الحكم. انظر النص كاملًا كما ورد في المصدر.';
      } else {
        title = 'موثّق: النص مطابق للمصدر';
        description =
            'توجد مطابقة مع المصدر الموجود في قاعدة المعرفة المستخدمة للتحقق.';
      }

      icon = Icons.verified_outlined;

      background = const Color(0xFFE8F1EA);
    } else if (status == 'يحتاج تصحيح' || status == 'يحتاج سياق') {
      if (issue == 'truncated') {
        title = 'مقتطع من سياقه';
        description =
            'النص الظاهر في الصورة جزء من نص أطول، وحُذف منه ما يغيّر الحكم أو فهمه (مثل شرط أو استثناء أو تفصيل). انظر النص كاملًا كما ورد في المصدر.';
      } else if (issue == 'altered') {
        title = issues.contains('truncated') ? 'محرّف ومقتطع' : 'محرّف عن أصله';
        description =
            'وجدنا أصل النص، لكن الصورة فيها كلمات أو حكم أو نسبة تختلف عن المصدر.';
        if (issues.contains('misattributed')) {
          description += ' كما نُسب إلى غير قائله أو مصدره.';
        }
        if (issues.contains('truncated')) {
          description += ' كما حُذف منه جزء من النص الأصلي.';
        }
      } else if (issue == 'misattributed') {
        title = 'منسوب لغير قائله';
        description =
            'النص موجود في المصدر، لكن الصورة نسبته إلى راوٍ أو قائل أو كتاب أو عالم غير المذكور في المصدر.';
        if (issues.contains('truncated')) {
          description += ' كما حُذف منه جزء من النص الأصلي.';
        }
      } else if (issue == 'not_authentic') {
        title = 'حديث لا يصح';
        description =
            'هذا النص مسجّل في المصدر ضمن الأحاديث المنتشرة التي لا تصح. وهذا بديل صحيح في معناه.';
      } else {
        title = 'يحتاج تصحيح';
        description =
            'تم العثور على أصل للنص، لكن النص الظاهر في الصورة يختلف عن المصدر أو يفتقد جزءًا من سياقه.';
      }

      icon = Icons.info_outline;

      background = const Color(0xFFF4EEDC);
    } else {
      title = 'غير موثّق';

      description =
          'لم نجد هذا النص في مصادرنا الموثوقة. وهذا لا يعني بالضرورة أنه خاطئ، وإنما لم نتمكن من توثيقه ضمن قاعدة المعرفة المستخدمة.';

      icon = Icons.search_off_outlined;

      background = const Color(0xFFF2E9E5);
    }

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: background,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: deepGreen, size: 27),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    color: darkText,
                    fontSize: 16,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 7),
                Text(
                  description,
                  style: const TextStyle(
                    color: mutedText,
                    fontSize: 13,
                    height: 1.65,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  static final RegExp _punct = RegExp('[،,.؛;:!؟?«»"\'()\\[\\]{}\\-ـ]');

  static String _cleanWord(String w) => w.replaceAll(_punct, '').trim();

  Widget _highlightedText(String text, Set<String> highlight, Color color) {
    const base = TextStyle(color: darkText, fontSize: 14, height: 1.8);
    if (highlight.isEmpty) return Text(text, style: base);
    final spans = <TextSpan>[];
    for (final part in text.split(RegExp(r'(?<=\s)|(?=\s)'))) {
      final bool hit = highlight.contains(_cleanWord(part));
      spans.add(TextSpan(
        text: part,
        style: hit
            ? TextStyle(
                color: color,
                fontWeight: FontWeight.w800,
                backgroundColor: color.withOpacity(0.10),
              )
            : null,
      ));
    }
    return Text.rich(TextSpan(children: spans), style: base);
  }

  Widget _buildTextSection({
    required String title,
    required String text,
    required IconData icon,
    Set<String> highlight = const {},
    Color highlightColor = darkText,
    String fullText = '',
  }) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFFE4DED2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: deepGreen, size: 21),
              const SizedBox(width: 8),
              Text(
                title,
                style: const TextStyle(
                  color: deepGreen,
                  fontSize: 15,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ],
          ),
          const SizedBox(height: 13),
          _highlightedText(text, highlight, highlightColor),
          if (fullText.isNotEmpty && fullText != text)
            Theme(
              data: ThemeData(dividerColor: Colors.transparent),
              child: ExpansionTile(
                tilePadding: EdgeInsets.zero,
                title: const Text(
                  'عرض النص كاملًا كما ورد في المصدر',
                  style: TextStyle(
                    color: deepGreen,
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                children: [
                  Align(
                    alignment: AlignmentDirectional.centerStart,
                    child: Text(
                      fullText,
                      style: const TextStyle(
                        color: mutedText,
                        fontSize: 13,
                        height: 1.8,
                      ),
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Future<void> _openUrl(BuildContext context, String url) async {
    final Uri? uri = Uri.tryParse(url);
    final bool opened =
        uri != null &&
        await launchUrl(uri, mode: LaunchMode.externalApplication);
    if (!opened && context.mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('تعذر فتح رابط المصدر')),
      );
    }
  }

  Widget _buildReferenceLine(Map ref) {
    final String name = ref['name']?.toString() ?? '';
    final String book = ref['book']?.toString() ?? '';
    final String page = ref['page']?.toString() ?? '';
    final String madhhab = ref['madhhab']?.toString() ?? '';
    final String note = ref['note']?.toString() ?? '';

    final List<String> details = [
      if (book.isNotEmpty) page.isNotEmpty ? '$book ($page)' : book,
      if (madhhab.isNotEmpty) madhhab,
      if (note.isNotEmpty) note,
    ];

    return Padding(
      padding: const EdgeInsets.only(bottom: 7),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.only(top: 7),
            child: Icon(Icons.circle, size: 5, color: lightGold),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text.rich(
              TextSpan(
                children: [
                  if (name.isNotEmpty)
                    TextSpan(
                      text: name,
                      style: const TextStyle(
                        color: lightGold,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  if (name.isNotEmpty && details.isNotEmpty)
                    const TextSpan(text: ' — '),
                  TextSpan(text: details.join(' · ')),
                ],
              ),
              style: TextStyle(
                color: Colors.white.withOpacity(0.85),
                fontSize: 13,
                height: 1.6,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSourceCard(
    BuildContext context,
    Map source, {
    String heading = 'المصدر',
  }) {
    final String scholar = source['scholar']?.toString() ?? '';

    final String title = source['title']?.toString() ?? '';

    final String url = source['url']?.toString() ?? '';

    final String narrator = source['narrator']?.toString() ?? '';

    final String grade = source['grade']?.toString() ?? '';

    final List<dynamic> references =
        source['references'] is List ? source['references'] as List : [];

    return Container(
      decoration: BoxDecoration(
        color: deepGreen,
        borderRadius: BorderRadius.circular(22),
      ),
      clipBehavior: Clip.antiAlias,
      child: Stack(
        children: [
          Positioned(
            left: -20,
            bottom: -25,
            child: SizedBox(
              width: 170,
              height: 150,
              child: IslamicPattern(opacity: 0.13, goldOnly: true),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(19),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      width: 43,
                      height: 43,
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.10),
                        borderRadius: BorderRadius.circular(13),
                      ),
                      child: const Icon(
                        Icons.menu_book_outlined,
                        color: lightGold,
                      ),
                    ),
                    const SizedBox(width: 11),
                    Text(
                      heading,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 17),
                if (scholar.isNotEmpty)
                  Text(
                    scholar,
                    style: const TextStyle(
                      color: lightGold,
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                if (title.isNotEmpty) ...[
                  const SizedBox(height: 7),
                  Text(
                    title,
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.85),
                      fontSize: 13,
                      height: 1.6,
                    ),
                  ),
                ],
                if (narrator.isNotEmpty) ...[
                  const SizedBox(height: 7),
                  Text(
                    'الراوي: $narrator',
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.85),
                      fontSize: 13,
                      height: 1.6,
                    ),
                  ),
                ],
                if (grade.isNotEmpty) ...[
                  const SizedBox(height: 7),
                  Text(
                    scholar.isNotEmpty
                        ? 'الدرجة: $grade (حكم $scholar)'
                        : 'الدرجة: $grade',
                    style: const TextStyle(
                      color: lightGold,
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                      height: 1.6,
                    ),
                  ),
                ],
                if (references.isNotEmpty) ...[
                  const SizedBox(height: 14),
                  Text(
                    'من العلماء المذكورين في المصدر:',
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.70),
                      fontSize: 12.5,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 8),
                  ...references.whereType<Map>().map(_buildReferenceLine),
                ],
                if (url.isNotEmpty && url != 'https://example.com') ...[
                  const SizedBox(height: 15),
                  OutlinedButton.icon(
                    onPressed: () => _openUrl(context, url),
                    icon: const Icon(Icons.open_in_new, size: 17),
                    label: const Text('عرض المصدر'),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: Colors.white,
                      side: BorderSide(color: gold.withOpacity(0.55)),
                    ),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}
