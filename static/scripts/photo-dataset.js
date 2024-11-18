let mediaStream = null;
let mediaStreamFace = null;
let countCapture = 0;
let cropCapture = false;

const socket = io.connect("http://" + document.domain + ":" + location.port);

socket.on("photo-dataset", function (data) {
  console.log(data);
  imgBody = "data:image/jpeg;base64," + data.frame;
  document.querySelector("#video").src = imgBody;
  if(data.status) {
    closeCamera();
    alert('Captura Finalizada')
  }
});

function startDetectPerson(camera) {
  fetch("/create-photo-dataset", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ camera_indices: [camera] }),
  })
    .then((response) => response.json())
    .catch((error) => {
      console.error("Error:", error);
    });
}

function stopDetectBody(camera, imgDefault = true) {
  if (imgDefault)
    document.getElementById("video").src = "static/images/camera.png";
  fetch("/stop-detect-body-face", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ camera_indices: [camera] }),
  })
    .then((response) => {
      return response.json();
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

document.getElementById("camera-button").addEventListener("click", (e) => {
  cropCapture = false
  startDetectPerson($("#cameraSelect :selected").val());
  openCamera();
});

document.querySelector('#btn-train').addEventListener('click', e => {
  e.preventDefault();
  $('#modalProgressbar').modal('show');
  const http = new XMLHttpRequest();
  http.open("POST", "/start-train");
  http.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      let data = this.responseText;
      data = JSON.parse(data);
      $('#modalProgressbar').modal('hide');
      showMessage(data.message);
    }
  }
  http.send();
  
})

document.getElementById("camera-close").addEventListener("click", (e) => {
  // stopDetectBody($("#cameraSelect :selected").val());
  closeCamera();
});

function openCamera() {
  const video = document.getElementById("video");

  document.getElementById("camera-close").style.display = "block";
  document.getElementById("camera-button").style.display = "none";
  video.style.display = "block";
}

function closeCamera() {
  document.getElementById("video").src = "static/images/camera.png";
  document.getElementById("camera-button").style.display = "block";
  document.getElementById("camera-close").style.display = "none";
}

function showMessage(message) {
  Swal.fire(message);
}
