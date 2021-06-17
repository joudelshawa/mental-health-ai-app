import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:provider/provider.dart';

import 'notifs.dart';

//main runs the launch screen
Future<void> main() async {
  runApp(SoQuoApp());
}

//app class
class SoQuoApp extends StatelessWidget {

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
        child: MaterialApp(
          home: HomePage(),
          debugShowCheckedModeBanner: false,
        ),
        providers: [
          ChangeNotifierProvider(create: (_) => NotificationService())
        ]);
  }
}

//app that runs from notifications
class FromNotifs extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MultiProvider(
        child: MaterialApp(
          home: PollPage(),
          debugShowCheckedModeBanner: false,
        ),
        providers: [
          ChangeNotifierProvider(create: (_) => NotificationService())
        ]);
  }
}

//homepage class
class HomePage extends StatefulWidget{
  HomePage();

  @override
  _HomePageState createState() => _HomePageState();
}

//app state class
class _HomePageState extends State<HomePage> {
  FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin = new FlutterLocalNotificationsPlugin();

  @override
  void initState() {
    Provider.of<NotificationService>(context, listen: false).initialize();
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: defaultAB,
      body: Center(
        child: Column(
          children: [
            Image.asset('assets/logo.png',
                height: 300.0,
                width: 300.0,
            ),
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
      floatingActionButton: Consumer<NotificationService>(
        builder: (context, model, _) =>
          FloatingActionButton(
            onPressed: () => model.instantNotification(),
            child: Icon(Icons.announcement_rounded),
            backgroundColor: darkAccent,
          ),
      ),
      backgroundColor: Colors.white,
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
      backgroundColor: Colors.white,
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
        backgroundColor: darkAccent,
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
              child: neutralButton
            ),
            SizedBox(height: 30),
            Row(
              children: [
                Column(
                  children: [
                    happyButton,
                    SizedBox(height: 30.0),
                    calmButton,
                    SizedBox(height: 30.0),
                    stressedButton,
                    SizedBox(height: 30.0),
                    angryButton,
                    SizedBox(height: 30.0)
                ]),
                SizedBox(width: 75.0),
                Column(
                  children: [
                    excitedButton,
                    SizedBox(height: 30.0),
                    sadButton,
                    SizedBox(height: 30.0),
                    owButton,
                    SizedBox(height: 30.0),
                    scaredButton,
                    SizedBox(height: 30.0)
                ]),
              ],
              mainAxisAlignment: MainAxisAlignment.center,
            ),
        ]),
      ),
      backgroundColor: Colors.white,
    );
  }
}

//styles
final ButtonStyle ovalButton = ElevatedButton.styleFrom(
  primary: lightAccent,
  fixedSize: Size(300, 64),
  padding: EdgeInsets.symmetric(horizontal: 16.0),
  shape: RoundedRectangleBorder(
    borderRadius: BorderRadius.circular(36.0),
  ),
);

final TextStyle titleStyle = TextStyle(
  fontWeight: FontWeight.w300,
  fontSize: 36.0,
  color: darkAccent,
);

final TextStyle buttonStyle = TextStyle(
  fontWeight: FontWeight.w300,
  fontSize: 30.0,
  color: Colors.white,
  fontStyle: FontStyle.italic,
);

final TextStyle labelStyle = TextStyle(
  fontSize: 18.0,
  fontStyle: FontStyle.italic,
  color: darkAccent,
);

final TextStyle bodyStyle = TextStyle(
  fontSize: 18.0,
  height: 1.5,
  color: darkAccent,
);

final AppBar defaultAB = AppBar(
  backgroundColor: darkAccent,
);

final Color base = new Color(0xff0abed5);
final Color darkAccent = new Color(0xff0ea0b2);
final Color lightAccent = new Color(0xFF2ee0f7);

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

//imagebuttons for emotions
final ImageButton neutralButton = new ImageButton(
    "assets/polling/neutral.png",
    "no change",
    50.0);

final ImageButton happyButton = new ImageButton(
    "assets/polling/happy.png",
    "happy",
    50.0);

final ImageButton excitedButton = new ImageButton(
    "assets/polling/excited.png",
    "excited",
    50.0);

final ImageButton calmButton = new ImageButton(
    "assets/polling/calm.png",
    "calm",
    50.0);

final ImageButton sadButton = new ImageButton(
    "assets/polling/sad.png",
    "sad",
    50.0);

final ImageButton stressedButton = new ImageButton(
    "assets/polling/stressed.png",
    "stressed",
    50.0);

final ImageButton owButton = new ImageButton(
    "assets/polling/overwhelmed.png",
    "overwhelmed",
    50.0);

final ImageButton angryButton = new ImageButton(
    "assets/polling/angry.png",
    "angry",
    50.0);

final ImageButton scaredButton = new ImageButton(
    "assets/polling/scared.png",
    "scared",
    50.0);


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
      ..color = lightAccent
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
