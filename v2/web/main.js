let video = document.getElementById("video");
let model;
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
const accessCamera = () => {
  navigator.mediaDevices
    .getUserMedia({
      video: { width: 94, height: 94 },
      audio: false,
    })
    .then((stream) => {
      video.srcObject = stream;
    });
};

const detectFaces = async () => {
  let example = tf.browser.fromPixels(video); 
  
  const smalImg = tf.image.resizeBilinear(example, [94, 94]);
const resized = tf.cast(smalImg, 'float32');
const t4d = tf.tensor4d(Array.from(resized.dataSync()),[1,94,94,3])


  
  const prediction = await model.predict(t4d, false);
  console.log("PREDICTION: " + prediction);

 // Using canvas to draw the video first

 ctx.drawImage(video, 0, 0, 500, 400);

/* prediction.forEach((predictions) => {
   
   // Drawing rectangle that'll detect the face
   ctx.beginPath();
   ctx.lineWidth = "4";
   ctx.strokeStyle = "yellow";
   ctx.rect(
     predictions.topLeft[0],
     predictions.topLeft[1],
     predictions.bottomRight[0] - predictions.topLeft[0],
     predictions.bottomRight[1] - predictions.topLeft[1]
   );
   // The last two arguments denotes the width and height
   // but since the blazeface models only returns the coordinates  
   // so we have to subtract them in order to get the width and height
   ctx.stroke();
 });*/
  
};

accessCamera();

// this event will be  executed when the video is loaded

video.addEventListener("loadeddata", async () => {
  model = await tf.loadLayersModel('http://localhost:8000/model/model1js/model.json');
  console.log(model);
  
  setInterval(detectFaces, 100);
});