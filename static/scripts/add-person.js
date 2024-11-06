let mediaStream = null;
let mediaStreamFace = null;
let countCapture = 0;
let cropCapture = false;

const socket = io.connect('http://' + document.domain + ':' + location.port);

socket.on('person_recognized', function (data) {
  console.log(data);
  if(data.type == 'face') {
    if(data.frame) {
      document.querySelector('#person').value = data.person;
    }
    document.querySelector('#video-face').src = 'data:image/jpeg;base64,' + data.frame;
  } else if(data.type == 'body') {
    if(!cropCapture)
      imgBody = 'data:image/jpeg;base64,' + data.body_person;
    if(data.status <= 0.96) {
      countCapture++;
      if(countCapture == 10) {
        stopDetectBody($('#cameraSelect :selected').val(), false);
        closeCamera(1);
        document.querySelector('#camera-save').style.display = 'block';
        cropCapture = true;
        imgBody = 'data:image/jpeg;base64,' + data.person_crop;
        // document.querySelector('#video').src = 'data:image/jpeg;base64,' + data.person_crop;
      }
    } else {
      countCapture = 0;
    }
    document.querySelector('#video').src = imgBody;
  }
});

function startTracking(camera) { 
    fetch('/start_tracking', {  
        method: 'POST',  
        headers: {  
            'Content-Type': 'application/json',  
        },
        body: JSON.stringify({ camera_indices: [camera] }),  
    })
    .then(response => response.json())  
    .then(data => {  
        console.log('Tracking started:', data);  
    })
    .catch((error) => {  
        console.error('Error:', error);  
    });
};

function startDetectBody(camera) { 
  fetch('/detect_body', {  
    method: 'POST',  
    headers: {  
      'Content-Type': 'application/json',  
    },
    body: JSON.stringify({ camera_indices: [camera] }),  
  })
  .then(response => response.json())  
  .catch((error) => {  
    console.error('Error:', error);  
  });
};


function stopDetectBody(camera, imgDefault = true) {
  if(imgDefault)  
    document.getElementById("video").src = "static/images/camera.png";
  fetch('/stop_detect_body', {  
      method: 'POST',  
      headers: {  
          'Content-Type': 'application/json',  
      },
      body: JSON.stringify({ camera_indices: [camera] }),  
  })
  .then(response => {
    return response.json();  
  })
  .catch((error) => {  
    console.error('Error:', error);  
  });
};

function stopTracking(camera) {  
    fetch('/stop_tracking', {  
        method: 'POST',  
        headers: {  
            'Content-Type': 'application/json',  
        },
        body: JSON.stringify({ camera_indices: [camera] }),  
    })
    .then(response => {
        console.log('EN MEDIO DE STOP');
        document.getElementById("video-face").src = "static/images/camera.png";
        return response.json();  
    })
    .catch((error) => {  
        console.error('Error:', error);  
    });
};

document.querySelector("#cameraSelect").addEventListener("change", (e) => {
  /* startCameraSelected(document.querySelector("#cameraSelect").value, 1);
  openCamera(1); */
  // startCameraSelected(document.querySelector("#cameraSelect").value, 1);
  startDetectBody($('#cameraSelect :selected').val());
  openCamera(1);
  document.querySelector('#camera-save').style.display = 'none';
});
document.querySelector("#cameraSelect-face").addEventListener("change", (e) => {
  startTracking($('#cameraSelect-face :selected').val());
  openCamera(2);
});

document.getElementById("camera-button-face").addEventListener("click", (e) => {
  startTracking($('#cameraSelect-face :selected').val());
  openCamera(2);
});
document.getElementById("camera-button").addEventListener("click", (e) => {
  startDetectBody($('#cameraSelect :selected').val());
  openCamera(1);
});
document.getElementById("camera-close-face").addEventListener("click", (e) => {
  stopTracking($('#cameraSelect :selected').val());
  closeCamera(2)
});
document.getElementById("camera-close").addEventListener("click", (e) => {
  stopDetectBody($('#cameraSelect :selected').val());
  closeCamera(1)
});

document.querySelector('#camera-save').addEventListener('click', (e) => {
    e.preventDefault();
    //validar que contenga el nombre del archivo
    if(document.querySelector('#video').getAttribute('src').length > 50 && document.querySelector('#person').value != '') {
      $('#modalProgressbar').modal('show');
      const http = new XMLHttpRequest();
      let formData = new FormData();
      formData.append('person', document.querySelector('#person').value.trim());
      formData.append('image', document.querySelector('#video').getAttribute('src'));
      http.open("POST", "/save-crop");
      http.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
          let data = this.responseText;
          data = JSON.parse(data);
          $('#modalProgressbar').modal('hide');
          document.querySelector('#person').value = '';
          showMessage(data.message);
          closeCamera(1);
        }
      }
      http.send(formData);
    } else {
      showMessage('Debe detectar el rostro !!!');
    }
});

$("#container-images").on("click", ".btn-remove", (e) => {
  e.target.parentElement.remove();
  if (document.querySelector("#container-images").childElementCount == 0)
    document.querySelector("#btn-upload-images").classList.add("d-none");
  else document.querySelector("#btn-upload-images").classList.remove("d-none");
});

function init() {
  // getSelectDevices();
}

function openCamera(type) {
  let aux = type == "1" ? "" : "-face";
  const video = document.getElementById("video");
  const videoFace = document.getElementById("video-face");

  document.getElementById("camera-close" + aux).style.display = "block";
  document.getElementById("camera-button" + aux).style.display = "none";
  if (type == "1") video.style.display = "block";
  else videoFace.style.display = "block";
}

function startCameraSelected(deviceId, type) {
  if(!deviceId) return;
  const video = document.getElementById("video");
  const videoFace = document.getElementById("video-face");
  let captureButton = document.getElementById("capture-button");
  let captureButtonFace = document.getElementById("capture-button-face");

  if (type == "1") {
    navigator.mediaDevices
      .getUserMedia({
        video: { deviceId: deviceId ? { exact: deviceId } : undefined },
        audio: false,
      })
      .then(function (stream) {
        mediaStream = stream;
        video.srcObject = stream;
        captureButton.style.display = "block";
      })
      .catch(function (error) {
        console.error("Error al acceder a la cámara: ", error);
        alert("Error al acceder a la cámara: ", error);
      });
  } else {
    navigator.mediaDevices
      .getUserMedia({
        video: { deviceId: deviceId ? { exact: deviceId } : undefined },
        audio: false,
      })
      .then(function (stream) {
        mediaStreamFace = stream;
        videoFace.srcObject = stream;
        captureButtonFace.style.display = "block";
      })
      .catch(function (error) {
        console.error("Error al acceder a la cámara: ", error);
        alert("Error al acceder a la cámara: ", error);
    });
  }
}

function closeCamera(type) {
  if(type == '1') {
    document.getElementById("video").src = "static/images/camera.png";
    document.getElementById("camera-button").style.display = "block";
    document.getElementById("camera-close").style.display = "none";
    // document.querySelector('#camera-save').style.display = 'none';
  } else {
    document.getElementById("camera-button-face").style.display = "block";
    document.getElementById("camera-close-face").style.display = "none";
  }
}

function showMessage(message) {
  Swal.fire(message);
}

init();
