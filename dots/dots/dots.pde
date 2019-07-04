import processing.video.*;
Movie mov;

int renderMode = 0;

Pose[][] poses;
int numPoses = 0;

void setup() {
  size(1280, 720);
  frameRate(25);
  smooth();
  String filename = "salsa";
  JSONArray json = loadJSONArray("../" + filename + ".JSON");
  poses = parseJsonToPoses(json);
  mov = new Movie(this, "/Users/mayank/ml-art/dots/" + filename + ".mp4");
  mov.play();
  stroke(255);
  background(0);
}

void draw() {
  //image(mov, 0, 0);
  //background(0);
  fill(0, 75);
  rect(0,0,width,height);
  float time = mov.time();
  int frameNum = int(time / 0.04);
  for (int i=0;i<numPoses;i++) {
    if (frameNum < poses[i].length) {
      Pose currentPose = poses[i][frameNum];
      if (currentPose.points.size() > 0) {
        if (renderMode == 0) {
          currentPose.renderDots();
        } else if (renderMode == 1) {
          currentPose.renderPoseNet();
        } else if (renderMode == 2) {
          currentPose.renderGeometry();
        } else if (renderMode == 3) {
          currentPose.renderFlashingLines(50);  
        } else if (renderMode == 4) {
          currentPose.renderGeometry();
          currentPose.renderFlashingLines(50);  
        } else if (renderMode == 3) {        
        }
      }
    }
  }
}

void keyPressed() {
  renderMode = key - 48;
  //print(int(key));
}
void movieEvent(Movie m) { 
  m.read(); 
} 
