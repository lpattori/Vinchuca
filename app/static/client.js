var el = x => document.getElementById(x);
var orig_img ;
function showPicker() {
    el("file-input").click();
}

function showPicked(input) {
    el("upload-label").innerHTML = input.files[0].name;
    var reader = new FileReader();
    reader.onload = function(e) {
        el("image-picked").src = e.target.result;
        orig_img = e.target.result;
        el("image-picked").className = "";
    };
    reader.readAsDataURL(input.files[0]);
    el("result-label").innerHTML = "" ;

}

function analyze() {
    var uploadFiles = el("file-input").files;
    if (uploadFiles.length !== 1) alert("Por favor seleccione una imágen para analizar!");

    el("analyze-button").innerHTML = "Analizando...";
    el("result-label").innerHTML = "" ;
    var xhr = new XMLHttpRequest();
    var loc = window.location;
    xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`,
        true);
    xhr.onerror = function() {
        alert(xhr.responseText);
    };
    xhr.onload = function(e) {
        if (this.readyState === 4) {
            var response = JSON.parse(e.target.responseText);
            txt = "<p>Clickee en las llamas para el mapa de calor de la categoría. Para restaurar clickee en la imágen</p>" ;
            for (i in response) {
                modelos = response [i];
                txt += "<p>" + modelos[0];
                modelo = modelos[1];
                for (j in modelo) {
                    clase = modelo[j];
                    txt += " " + clase[0] + " " + clase[1] +
                        "<button class='btn' " + "id = heat" + i + j +
                        " onclick='show_heatmap(this, " + i + ", " +
                        clase[2] + (j==0 ? ", true":", false") +
                        ")'>🔥</button>" ;
                }
                txt += "</p>";
            }
            el("result-label").innerHTML = txt ;
        }
        el("analyze-button").innerHTML = "Analizar";
    };

    var fileData = new FormData();
    fileData.append("file", uploadFiles[0]);
    xhr.send(fileData);
}

function show_heatmap(element, learner, clase, first) {
    var xhr = new XMLHttpRequest();
    var loc = window.location;
    var respuesta ;
    if (element.dataset.src) {
        highlight(element);
        var img = document.querySelector( "#image-picked" );
        img.src = element.dataset.src;
        return
    }
    xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/heat/`,
        true);
    xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");

    xhr.responseType = 'arraybuffer';

    xhr.onerror = function() {
        alert(xhr.responseText);
    };
    xhr.onload = function(e) {
        if (this.readyState === 4) {
            var arrayBufferView = new Uint8Array( e.target.response );
            var blob = new Blob( [ arrayBufferView ], { type: "image/jpeg" } );
            var urlCreator = window.URL || window.webkitURL;
            var imageUrl = urlCreator.createObjectURL( blob );
            var img = document.querySelector( "#image-picked" );
            highlight(element);
            img.src = imageUrl;
            element.dataset.src = imageUrl;
        }
    };
    if (!first) {first = false}
    var calor = {learner: learner, clase: clase, first: first};
    var string = JSON.stringify(calor);
    xhr.send(string);
}

function restore_image() {
    highlight();
    el("image-picked").src = orig_img ;
}

var buttonClicked = null;

function highlight(element) {
  if (buttonClicked != null) {
      buttonClicked.style.background = "white";
      buttonClicked.style.color = "black";
  }
  if (element) {
      buttonClicked = element;
      buttonClicked.style.background = "red";
      buttonClicked.style.color = "white";
  }
}