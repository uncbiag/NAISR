window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_STARMAN_HAND_BASE = "./static/interpolation/starman/hand";
var INTERP_STARMAN_LEG_BASE = "./static/interpolation/starman/leg";

var INTERP_AIRWAY_AGE_BASE = "./static/interpolation/airway/age";
var INTERP_AIRWAY_WEIGHT_BASE = "./static/interpolation/airway/weight";
var INTERP_AIRWAY_SEX_BASE = "./static/interpolation/airway/sex";

var INTERP_ADNI_AGE_BASE = "./static/interpolation/adni/age";
var INTERP_ADNI_AD_BASE = "./static/interpolation/adni/AD";
var INTERP_ADNI_EDU_BASE = "./static/interpolation/adni/edu";
var INTERP_ADNI_SEX_BASE = "./static/interpolation/adni/sex";

var NUM_INTERP_FRAMES = 20;
var interp_starman_hand_images = [];
var interp_starman_leg_images = [];

var interp_airway_age_images = [];
var interp_airway_weight_images = [];
var interp_airway_sex_images = [];

var interp_adni_age_images = [];
var interp_adni_AD_images = [];
var interp_adni_edu_images = [];
var interp_adni_sex_images = [];


function preloadImages() {

    for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
        var path = INTERP_STARMAN_HAND_BASE + '/' + String(i).padStart(6, '0') + '.png';
        interp_starman_hand_images[i] = new Image();
        interp_starman_hand_images[i].src = path;
    }

    for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
        var path = INTERP_STARMAN_LEG_BASE + '/' + String(i).padStart(6, '0') + '.png';
        interp_starman_leg_images[i] = new Image();
        interp_starman_leg_images[i].src = path;
    }

    for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_AIRWAY_AGE_BASE + '/' + String(i).padStart(6, '0') + '.png';
    interp_airway_age_images[i] = new Image();
    interp_airway_age_images[i].src = path;
    }

    for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_AIRWAY_WEIGHT_BASE + '/' + String(i).padStart(6, '0') + '.png';
    interp_airway_weight_images[i] = new Image();
    interp_airway_weight_images[i].src = path;
    }

    for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_AIRWAY_SEX_BASE + '/' + String(i).padStart(6, '0') + '.png';
    interp_airway_sex_images[i] = new Image();
    interp_airway_sex_images[i].src = path;
    }

    for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_ADNI_AGE_BASE + '/' + String(i).padStart(6, '0') + '.png';
    interp_adni_age_images[i] = new Image();
    interp_adni_age_images[i].src = path;
    }

    for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_ADNI_AD_BASE + '/' + String(i).padStart(6, '0') + '.png';
    interp_adni_AD_images[i] = new Image();
    interp_adni_AD_images[i].src = path;
    }

    for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_ADNI_EDU_BASE + '/' + String(i).padStart(6, '0') + '.png';
    interp_adni_edu_images[i] = new Image();
    interp_adni_edu_images[i].src = path;
    }
    for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_ADNI_SEX_BASE + '/' + String(i).padStart(6, '0') + '.png';
    interp_adni_sex_images[i] = new Image();
    interp_adni_sex_images[i].src = path;
  }
}


function setStarmanHand(i) {
  var image = interp_starman_hand_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#wrapper-starman-hand').empty().append(image);
}

function setStarmanLeg(i) {
  var image = interp_starman_leg_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#wrapper-starman-leg').empty().append(image);
}


function setAirwayAge(i) {
  var image = interp_airway_age_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#wrapper-airway-age').empty().append(image);
}

function setAirwayWeight(i) {
  var image = interp_airway_weight_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#wrapper-airway-weight').empty().append(image);
}

function setAirwaySex(i) {
  var image = interp_airway_sex_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#wrapper-airway-sex').empty().append(image);
}

function setADNIAge(i) {
  var image = interp_adni_age_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#wrapper-adni-age').empty().append(image);
}

function setADNIAD(i) {
  var image = interp_adni_AD_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#wrapper-adni-AD').empty().append(image);
}


function setADNIEdu(i) {
  var image = interp_adni_edu_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#wrapper-adni-edu').empty().append(image);
}

function setADNISex(i) {
  var image = interp_adni_sex_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#wrapper-adni-sex').empty().append(image);
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadImages();
    $('#starman-hand-slider').on('input', function(event) {
      setStarmanHand(this.value);
    });
    setStarmanHand(0);
    $('#starman-hand-slider').prop('max', NUM_INTERP_FRAMES - 1);

    $('#starman-leg-slider').on('input', function(event) {
      setStarmanLeg(this.value);
    });
    setStarmanLeg(0);
    $('#starman-leg-slider').prop('max', NUM_INTERP_FRAMES - 1);



    $('#airway-age-slider').on('input', function(event) {
      setAirwayAge(this.value);
    });
    setAirwayAge(0);
    $('#airway-age-slider').prop('max', NUM_INTERP_FRAMES - 1);

    $('#airway-weight-slider').on('input', function(event) {
      setAirwayWeight(this.value);
    });
    setAirwayWeight(0);
    $('#airway-weight-slider').prop('max', NUM_INTERP_FRAMES - 1);


    $('#airway-sex-slider').on('input', function(event) {
      setAirwaySex(this.value);
    });
    setAirwaySex(0);
    $('#airway-sex-slider').prop('max', NUM_INTERP_FRAMES - 1);


    $('#ADNI-age-slider').on('input', function(event) {
      setADNIAge(this.value);
    });
    setADNIAge(0);
    $('#ADNI-age-slider').prop('max', NUM_INTERP_FRAMES - 1);

    $('#ADNI-AD-slider').on('input', function(event) {
      setADNIAD(this.value);
    });
    setADNIAD(0);
    $('#ADNI-AD-slider').prop('max', NUM_INTERP_FRAMES - 1);

    $('#ADNI-edu-slider').on('input', function(event) {
      setADNIEdu(this.value);
    });
    setADNIEdu(0);
    $('#ADNI-edu-slider').prop('max', NUM_INTERP_FRAMES - 1);

    $('#ADNI-sex-slider').on('input', function(event) {
      setADNISex(this.value);
    });
    setADNISex(0);
    $('#ADNI-Sex-slider').prop('max', NUM_INTERP_FRAMES - 1);


    bulmaSlider.attach();

})
