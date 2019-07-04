class Lowpass {
  float alpha = 0.2;
  float y = -1;
  
  float process(float input) {
    if (y == -1) {
      y = input;
    }
    y = alpha * input + (1-alpha)*y;
    return y;
  }
}

class LowpassPoint {
  Lowpass x = new Lowpass();
  Lowpass y = new Lowpass();
  
  Point process(Point input) {
    return new Point(x.process(input.x), y.process(input.y));
  } 
}
