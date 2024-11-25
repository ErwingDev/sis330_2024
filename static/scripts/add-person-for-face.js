let mediaStream = null;
let mediaStreamFace = null;
let countCapture = 0;
let cropCapture = false;

const socket = io.connect("http://" + document.domain + ":" + location.port);

socket.on("detect-face-body", function (data) {
  console.log(data);
  document.querySelector('#person').value = data.name;
  if (!cropCapture) imgBody = "data:image/jpeg;base64," + data.frame;
  if (data.status <= 0.96) {
    countCapture++;
    // if (countCapture == 10) {
    if (countCapture == 7) {
      stopDetectBody($("#cameraSelect :selected").val(), false);
      closeCamera();
      document.querySelector("#camera-save").style.display = "block";
      cropCapture = true;
      imgBody = "data:image/jpeg;base64," + data.person_crop;
    }
  } else {
    countCapture = 0;
  }
  document.querySelector("#video").src = imgBody;
});

function startDetectBody(camera) {
  let timeRemaining = 5
  const intervalId = setInterval(function() {
    timeRemaining--;
    document.querySelector('#timer').textContent = timeRemaining;
    
    if (timeRemaining === 0) {
      document.querySelector('#timer').textContent = '';
      clearInterval(intervalId);
      fetch("/detect-body-with-face", {
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
  }, 1000); 
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

document.querySelector("#cameraSelect").addEventListener("change", (e) => {
  startDetectBody($("#cameraSelect :selected").val());
  openCamera();
  document.querySelector("#camera-save").style.display = "none";
});

document.getElementById("camera-button").addEventListener("click", (e) => {
  cropCapture = false
  startDetectBody($("#cameraSelect :selected").val());
  openCamera();
});

document.getElementById("camera-close").addEventListener("click", (e) => {
  stopDetectBody($("#cameraSelect :selected").val());
  closeCamera();
});

document.querySelector("#camera-save").addEventListener("click", (e) => {
  e.preventDefault();
  //validar que contenga el nombre del archivo
  if (
    document.querySelector("#video").getAttribute("src").length > 50 &&
    document.querySelector("#person").value != ""
  ) {
    $("#modalProgressbar").modal("show");
    const http = new XMLHttpRequest();
    let formData = new FormData();
    formData.append("person", document.querySelector("#person").value.trim());
    formData.append(
      "image",
      document.querySelector("#video").getAttribute("src")
    );
    http.open("POST", "/save-crop");
    http.onreadystatechange = function () {
      if (this.readyState == 4 && this.status == 200) {
        let data = this.responseText;
        data = JSON.parse(data);
        $("#modalProgressbar").modal("hide");
        document.querySelector("#person").value = "";
        showMessage(data.message);
        closeCamera();
      }
    };
    http.send(formData);
  } else {
    showMessage("Debe detectar el rostro !!!");
  }
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
