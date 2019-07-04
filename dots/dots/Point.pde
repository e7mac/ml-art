class Point {
  float x;
  float y;
   
  Point(float x, float y) {
    this.x = x;
    this.y = y;
  }  

  Point(JSONObject object) {
    this.x = object.getFloat("x");
    this.y = object.getFloat("y");
  }  

}

void line(Point p1, Point p2) {
  line(p1.x, p1.y, p2.x, p2.y);
}
