package main

import (
	"embed"
	"voice-tts/desktop"
)

//go:embed all:frontend/dist
var assets embed.FS

func main() {
	desktop.Main(assets)
}
