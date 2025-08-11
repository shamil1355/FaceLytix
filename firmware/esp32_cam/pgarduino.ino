#include <WiFi.h>
#include <WebServer.h>
#include <esp32cam.h>
#include "esp_camera.h"  // Required for sensor_t and flash control

const char* WIFI_SSID = "Redmi Note 11";
const char* WIFI_PASS = "24682468";

WebServer server(80);

static auto loRes = esp32cam::Resolution::find(320, 240);
static auto midRes = esp32cam::Resolution::find(350, 530);
static auto hiRes = esp32cam::Resolution::find(800, 600);

#define FLASH_LED_PIN 4  // Built-in flash LED (AI-Thinker ESP32-CAM)

void serveJpg(bool flash)
{
  // Enable flash (if required)
  if (flash) {
    sensor_t* s = esp_camera_sensor_get();
    if (s != nullptr) {
      s->set_ledc(s, 1); // Turn ON Flash LED
    }
  }

  auto frame = esp32cam::capture();
  if (frame == nullptr) {
    Serial.println("CAPTURE FAIL");
    server.send(503, "", "");
    return;
  }
  Serial.printf("CAPTURE OK %dx%d %db\n", frame->getWidth(), frame->getHeight(),
                static_cast<int>(frame->size()));

  server.setContentLength(frame->size());
  server.send(200, "image/jpeg");
  WiFiClient client = server.client();
  frame->writeTo(client);

  // Disable flash after capture
  if (flash) {
    sensor_t* s = esp_camera_sensor_get();
    if (s != nullptr) {
      s->set_ledc(s, 0); // Turn OFF Flash LED
    }
  }
}

void handleJpgLo() {
  if (!esp32cam::Camera.changeResolution(loRes)) {
    Serial.println("SET-LO-RES FAIL");
  }
  serveJpg(false);
}

void handleJpgMid() {
  if (!esp32cam::Camera.changeResolution(midRes)) {
    Serial.println("SET-MID-RES FAIL");
  }
  serveJpg(false);
}

void handleJpgHi()
