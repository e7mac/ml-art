HashMap<String, Point> jsonToMap(JSONArray keypoints) {
  HashMap<String, Point> points = new HashMap();
  for (int i=0; i< keypoints.size();i++) {
    JSONObject point = keypoints.getJSONObject(i).getJSONObject("position");
    Point p = new Point(point);
    String part = keypoints.getJSONObject(i).getString("part");
    points.put(part, p);
  }
  return points;
}

Pose[][] parseJsonToPoses(JSONArray json) {
    //numPoses = json.getJSONObject(int(random(json.size()))).getJSONArray("poses").size();
    numPoses = 1;
    print(numPoses);
    Pose[][] ps = new Pose[numPoses][json.size()];
    HashMap<String, LowpassPoint> lpPart = new HashMap();
    for (int i=0;i<numPoses;i++) {
      Pose previousPose = null;
      for (int j = 0; j < json.size() ; j++) {
        JSONArray points = json.getJSONObject(j).getJSONArray("poses");
        if (points.size() > 0 && i < points.size()) {
          JSONArray keypoints = points.getJSONObject(i).getJSONArray("keypoints");
          HashMap<String, Point> map = jsonToMap(keypoints); 
          Object[] parts = map.keySet().toArray();
          for (int k=0;k<parts.length;k++) {
            String part = (String) parts[k];
            if (lpPart.get(part) == null) {
              lpPart.put(part, new LowpassPoint());
            }
            LowpassPoint lp = lpPart.get(part);
            map.put(part, lp.process(map.get(part)));
          }
           
          Pose pose = new Pose(map);
          ps[i][j] = pose;
          previousPose = pose;
        } else if (previousPose != null) {
          ps[i][j] = previousPose;
        } else {
          ps[i][j] = new Pose();
        }
      }
    }
    return ps;
}
