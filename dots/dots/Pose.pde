class Pose {
  HashMap<String, Point> points = new HashMap();
  Point nose;
  Point leftEye;
  Point rightEye;
  Point leftEar;
  Point rightEar;
  Point leftShoulder;
  Point rightShoulder;
  Point leftElbow;
  Point rightElbow;
  Point leftWrist;
  Point rightWrist;
  Point leftHip;
  Point rightHip;
  Point leftKnee;
  Point rightKnee;
  Point leftAnkle;
  Point rightAnkle; 

  Pose() {
  }

  Pose(HashMap<String, Point> points) {
     this.points = points;
     refreshPoints();
  }
  
  void refreshPoints() {
     this.nose = points.get("nose");
     this.leftEye = points.get("leftEye");
     this.rightEye = points.get("rightEye");
     this.leftEar = points.get("leftEar");
     this.rightEar = points.get("rightEar");
     this.leftShoulder = points.get("leftShoulder");
     this.rightShoulder = points.get("rightShoulder");
     this.leftElbow = points.get("leftElbow");
     this.rightElbow = points.get("rightElbow");
     this.leftWrist = points.get("leftWrist");
     this.rightWrist = points.get("rightWrist");
     this.leftHip = points.get("leftHip");
     this.rightHip = points.get("rightHip");
     this.leftKnee = points.get("leftKnee");
     this.rightKnee = points.get("rightKnee");
     this.leftAnkle = points.get("leftAnkle");
     this.rightAnkle = points.get("rightAnkle");    
  }
  
  void renderPoseNet() {
    renderDots();
    line(rightElbow, rightWrist);
    line(rightElbow, rightShoulder);
    
    line(leftElbow, leftWrist);
    line(leftElbow, leftShoulder);
    
    fill(0,0);
    quad(leftShoulder.x, leftShoulder.y, rightShoulder.x, rightShoulder.y, rightHip.x, rightHip.y, leftHip.x, leftHip.y);

    line(rightHip, rightKnee);
    line(rightKnee, rightAnkle);
    
    line(leftHip, leftKnee);
    line(leftKnee, leftAnkle);
  }

  void renderFlashingLines(int n) {
    Object[] keys = this.points.keySet().toArray();
    for (int i=0;i<n;i++) {
      int i1 = int(random(points.size() - 1));
      int i2 = int(random(points.size() - 1));
      Point p1 = this.points.get(keys[i1]);
      Point p2 = this.points.get(keys[i2]);
      line(p1, p2);
    }
  }

  void renderGeometry() {
    line(nose, rightWrist);
    line(nose, leftWrist);

    line(nose, rightShoulder);
    line(nose, leftShoulder);

    line(nose, rightElbow);
    line(nose, leftElbow);

    line(rightHip, rightElbow);
    line(leftHip, leftElbow);

  line(rightHip, leftHip);

    line(rightHip, rightKnee);
    line(rightKnee, rightAnkle);
    
    line(leftHip, leftKnee);
    line(leftKnee, leftAnkle);

    line(rightShoulder, leftShoulder);

    line(leftKnee, rightAnkle);
    line(leftKnee, rightKnee);
    line(leftHip, rightKnee);
    
    line(leftHip, leftWrist);
    line(rightHip, rightWrist);

    line(leftKnee, leftWrist);
    line(rightKnee, rightWrist);

    line(rightAnkle, rightWrist);
    line(leftAnkle, leftWrist);
    
    line(rightAnkle, leftAnkle);
    
    line(rightShoulder, leftHip);
    line(leftShoulder, leftHip);
    
    line(rightElbow, rightShoulder);
    line(leftElbow, leftShoulder);

  }

  void renderDots() {
    for (String part : points.keySet()) {
      Point p = points.get(part);
      fill(255);
      ellipse(p.x, p.y, 5, 5);
    }
  }

}
