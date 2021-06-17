import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:provider/provider.dart';

import 'notifs.dart';

//main runs the launch screen
Future<void> main() async {
  runApp(MyApp());
}

//lol idk
class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MultiProvider(
        child: MaterialApp(
          theme: ThemeData(fontFamily: 'Monteserat'),
          home: SoQuoApp(),
          debugShowCheckedModeBanner: false,
        ),
        providers: [
          ChangeNotifierProvider(create: (_) => NotificationService())
        ]);
  }
}

//app class
class SoQuoApp extends StatefulWidget{
  SoQuoApp();

  @override
  _SoQuoAppState createState() => _SoQuoAppState();
}

//app state class
class _SoQuoAppState extends State<SoQuoApp> {
  FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin = new FlutterLocalNotificationsPlugin();

  @override
  void initState() {
    Provider.of<NotificationService>(context, listen: false).initialize();
    super.initState();
  }

  @override
  Widget build(BuildContext context)
  {
    return Scaffold(
      body: Consumer<NotificationService>(
        builder: (context, model, _) =>
        Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton(
                onPressed: () => model.instantNotification(),
                child: Text('Instant Notification')
              ),
            ]
        )
      )
    );
  }
}

//homepage
class HomePage extends StatelessWidget
{
  HomePage();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: defaultAB,
      body: Center(
        child: Column(
          children: [
            Image.asset('assets/logo_placeholder.jpg'),
            SizedBox(height: 30.0),
            ElevatedButton(
              style: ovalButton,
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => QuotientPage()),
                );
              },
              child: Text("Log in", style: buttonStyle)
            ),
            SizedBox(height: 30.0),
            ElevatedButton(
              style: ovalButton,
              onPressed: () {},
              child: Text("Help", style: buttonStyle)
            )],
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
        ),
      ),
      backgroundColor: bg,
    );
  }
}

//quotient page
class QuotientPage extends StatelessWidget {
  QuotientPage();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: defaultAB,
      body: Column(
        children: [
          Stack(
            children: [
              Center(
                child: CustomPaint(
                  painter: new RingPainter(256, 256, Offset(0.0, 0.0),
                  1.5*3.14, 2*3.14*.863),
                ),
              ),
              Center(
                child: Text(
                  "86.3%",
                  style: titleStyle,
                ),
              ),
            ],
            alignment: Alignment(0.0, 0.0),
          ),
          SizedBox(height: 164),
          Container(
            child: Text(
              "Today's topics:",
              style: titleStyle,
            ),
            alignment: Alignment(-0.45, 0),
          ),
          SizedBox(height: 10),
          Container(
            child: Text(
              "1. example topic\n2. example topic\n3. example topic",
              style: bodyStyle,
            ),
            alignment: Alignment(-0.45, 0),
          ),
        ],
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.center,
      ),
      bottomNavigationBar: BottomNavigationBar(
        items: [
          BottomNavigationBarItem(
            icon: Icon(Icons.today_rounded),
            label: "Today"
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.settings_rounded),
            label: "Settings"
          ),
        ],
      ),
      backgroundColor: bg,
    );
  }
}

//poll screen
class PollPage extends StatelessWidget {
  PollPage();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: ab,
      ),
      body: Center(
        child: Column(
          children: [
            SizedBox(
              height: 30,
            ),
            Text("How do you feel?",
            style: titleStyle),
            SizedBox(
              height: 30,
            ),
            Center(
              child: ImageButton(
                "assets/polling/neutral.png",
                "no change",
                50.0
              ),
            ),
          ]
        ),
      ),
      backgroundColor: bg,
    );
  }
}

//styles
final ButtonStyle ovalButton = ElevatedButton.styleFrom(
  primary: accent,
  fixedSize: Size(300, 64),
  padding: EdgeInsets.symmetric(horizontal: 16.0),
  shape: RoundedRectangleBorder(
    borderRadius: BorderRadius.circular(36.0),
  ),
);

final TextStyle titleStyle = TextStyle(
  fontWeight: FontWeight.w300,
  fontSize: 36.0,
  color: Colors.black54,
);

final TextStyle buttonStyle = TextStyle(
  fontWeight: FontWeight.w300,
  fontSize: 30.0,
  color: Colors.black54,
  fontStyle: FontStyle.italic,
);

final TextStyle labelStyle = TextStyle(
  fontSize: 18.0,
  fontStyle: FontStyle.italic,
  color: Colors.black54,
);

final TextStyle bodyStyle = TextStyle(
  fontSize: 18.0,
  height: 1.5,
  color: Colors.black54,
);

final AppBar defaultAB = AppBar(
  backgroundColor: ab,
);

final Color bg = new Color(0xFF7FC1B2);
final Color ab = new Color(0xFF3C9378);
final Color accent = new Color(0xFF70E2C4);

//image with label
class ImageLabel extends StatelessWidget {
  final String image, label;
  final double size;

  ImageLabel(this.image, this.label, this.size);

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Image.asset(image, height: size, width: size),
        SizedBox(height: 5),
        Text(label, style: labelStyle)
      ],
    );
  }
}

//imagelabel button
class ImageButton extends StatelessWidget {
  final String image, label;
  final double size;

  ImageButton(this.image, this.label, this.size);

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
        onTap: () {
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => QuotientPage()),
          );
        },
        child: ImageLabel(image, label, size),
    );
  }
}

//ring shape painter
class RingPainter extends CustomPainter {
  double h = 0.0, w = 0.0, start = 0.0, sweep = 0.0;
  Offset c = Offset(0.0, 0.0);

  RingPainter(double height, double width, Offset center,
      double startAngle, double sweepAngle) {
    h = height;
    w = width;
    c = center;
    start = startAngle;
    sweep = sweepAngle;
  }

  @override
  void paint(Canvas canvas, Size size) {
    Paint paint = Paint()
      ..color = accent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 36.0
      ..strokeCap = StrokeCap.round;

    Path path = Path();

    path.arcTo(Rect.fromCenter(
        height: h,
        width: w,
        center: c),
      start, sweep, true);

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
