// Trigger rebuild to reload python server with new clone voice gender option support (Ryan/Serena)
package desktop

// Trigger dev rebuild to reload python server with new model downloader and hf_downloader integration
import (
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/wailsapp/wails/v3/pkg/application"
)

var assets embed.FS

var wailsApp *application.App
var pythonPort = 5050
var serverScriptPath = "backend/server.py"

func getFreePort() (int, error) {
	addr, err := net.ResolveTCPAddr("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	l, err := net.ListenTCP("tcp", addr)
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

type PythonServerManager struct {
	cmd *exec.Cmd
}

func NewPythonServerManager() *PythonServerManager {
	return &PythonServerManager{}
}

func setupProductionEnvironment() {
	execPath, err := os.Executable()
	if err != nil {
		return
	}
	dir := filepath.Dir(execPath)
	// Check if running inside macOS app bundle (Contents/MacOS/Tuti)
	if strings.HasSuffix(dir, "Contents/MacOS") {
		resourcesDir := filepath.Join(filepath.Dir(dir), "Resources")
		bundleServerPath := filepath.Join(resourcesDir, "server.py")
		if _, err := os.Stat(bundleServerPath); err == nil {
			// Set the global script path to the absolute path in Resources
			serverScriptPath = bundleServerPath

			// Change working directory to a writeable path in Application Support
			home, err := os.UserHomeDir()
			if err == nil {
				appSupport := filepath.Join(home, "Library", "Application Support", "Tuti")
				_ = os.MkdirAll(appSupport, 0755)
				_ = os.Chdir(appSupport)
				log.Printf("[Go Backend] Packaged App detected. Working directory changed to: %s", appSupport)
				log.Printf("[Go Backend] Python server script path set to: %s", serverScriptPath)
			}
		}
	}
}

func (m *PythonServerManager) Start() error {
	// Locate Python interpreter. 
	// Default to "python3", but prioritize Conda python on Mac if it exists.
	pythonPath := "python3"
	if runtime.GOOS == "darwin" {
		condaPath := "/opt/homebrew/Caskroom/miniconda/base/bin/python3"
		if _, err := os.Stat(condaPath); err == nil {
			pythonPath = condaPath
		}
	}

	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get current directory: %v", err)
	}

	if _, err := os.Stat(serverScriptPath); err != nil {
		return fmt.Errorf("server.py not found: %v", err)
	}

	log.Printf("[Go Backend] Starting Python server using %s...", pythonPath)
	m.cmd = exec.Command(pythonPath, serverScriptPath)
	m.cmd.Dir = cwd
	m.cmd.Stdout = os.Stdout
	m.cmd.Stderr = os.Stderr

	err = m.cmd.Start()
	if err != nil {
		return fmt.Errorf("failed to start Python subprocess: %v", err)
	}

	log.Printf("[Go Backend] Python server started with PID %d", m.cmd.Process.Pid)
	return nil
}

func (m *PythonServerManager) Stop() {
	if m.cmd != nil && m.cmd.Process != nil {
		log.Printf("[Go Backend] Stopping Python server (PID %d)...", m.cmd.Process.Pid)
		err := m.cmd.Process.Kill()
		if err != nil {
			log.Printf("[Go Backend] Failed to kill Python server: %v", err)
		} else {
			log.Printf("[Go Backend] Python server stopped successfully.")
		}
	}
}

func writeJSON(w http.ResponseWriter, status int, val interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(val)
}

func writeRawJSON(w http.ResponseWriter, status int, rawJSON string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write([]byte(rawJSON))
}

func handleListTimbres(w http.ResponseWriter, r *http.Request) {
	list, err := listTimbres()
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	if list == nil {
		list = []Timbre{}
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{"timbres": list})
}

func handleSaveTimbre(w http.ResponseWriter, r *http.Request) {
	var t Timbre
	err := json.NewDecoder(r.Body).Decode(&t)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "Invalid JSON body"})
		return
	}
	if t.Name == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "Timbre name is required"})
		return
	}
	err = saveTimbre(t)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{"saved": true, "timbre": t})
}

func handleDeleteTimbre(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name string `json:"name"`
	}
	name := r.URL.Query().Get("name")
	if name == "" {
		_ = json.NewDecoder(r.Body).Decode(&req)
		name = req.Name
	}
	if name == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "Timbre name is required"})
		return
	}
	err := deleteTimbre(name)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{"deleted": true})
}

func handleListProjects(w http.ResponseWriter, r *http.Request) {
	list, err := listProjects()
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	if list == nil {
		list = []Project{}
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{"projects": list})
}

func handleLoadProject(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID int `json:"id"`
	}
	var id int
	idStr := r.URL.Query().Get("id")
	if idStr != "" {
		fmt.Sscanf(idStr, "%d", &id)
	}
	if id == 0 {
		_ = json.NewDecoder(r.Body).Decode(&req)
		id = req.ID
	}
	if id == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "Project ID is required"})
		return
	}
	err := setActiveProjectID(id)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	stateJSON, err := getProject(id)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeRawJSON(w, http.StatusOK, stateJSON)
}

func handleDeleteProject(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID int `json:"id"`
	}
	var id int
	idStr := r.URL.Query().Get("id")
	if idStr != "" {
		fmt.Sscanf(idStr, "%d", &id)
	}
	if id == 0 {
		_ = json.NewDecoder(r.Body).Decode(&req)
		id = req.ID
	}
	if id == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "Project ID is required"})
		return
	}
	activeID, _ := getActiveProjectID()
	if activeID == id {
		_ = setActiveProjectID(0)
	}
	err := deleteProject(id)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	// Clean up project-specific workspace directory from disk
	cwd, errCwd := os.Getwd()
	if errCwd == nil {
		projectDir := filepath.Join(cwd, "pipeline_work", fmt.Sprintf("project_%d", id))
		if _, errStat := os.Stat(projectDir); errStat == nil {
			_ = os.RemoveAll(projectDir)
		}
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{"deleted": true})
}

func handleResetProject(w http.ResponseWriter, r *http.Request) {
	err := setActiveProjectID(0)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{"reset": true})
}

func handleProjectState(w http.ResponseWriter, r *http.Request) {
	activeID, err := getActiveProjectID()
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	if activeID > 0 {
		stateJSON, err := getProject(activeID)
		if err == nil && stateJSON != "" {
			writeRawJSON(w, http.StatusOK, stateJSON)
			return
		}
	}

	// Fallback to most recent
	stateJSON, err := getMostRecentProjectState()
	if err == nil && stateJSON != "" {
		writeRawJSON(w, http.StatusOK, stateJSON)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{"initialized": false})
}

func handleSelectFolder(w http.ResponseWriter, r *http.Request) {
	if wailsApp == nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "Wails application not initialized"})
		return
	}

	dialog := wailsApp.Dialog.OpenFile().
		CanChooseDirectories(true).
		CanChooseFiles(false).
		CanCreateDirectories(true).
		SetTitle("Select Export Directory")

	selectedPath, err := dialog.PromptForSingleSelection()
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"path": selectedPath})
}

func handleServeAudio(w http.ResponseWriter, r *http.Request) {
	pathParam := r.URL.Query().Get("path")
	if pathParam == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "Missing path parameter"})
		return
	}

	cwd, err := os.Getwd()
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "Failed to get working directory"})
		return
	}

	basePath := filepath.Join(cwd, "pipeline_work")
	absBasePath, err := filepath.Abs(basePath)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "Failed to resolve base path"})
		return
	}

	var targetPath string
	if filepath.IsAbs(pathParam) {
		targetPath = filepath.Clean(pathParam)
	} else {
		targetPath = filepath.Clean(filepath.Join(cwd, pathParam))
	}

	absTargetPath, err := filepath.Abs(targetPath)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "Invalid path parameter"})
		return
	}

	rel, err := filepath.Rel(absBasePath, absTargetPath)
	if err != nil || strings.HasPrefix(rel, "..") || rel == "." {
		writeJSON(w, http.StatusForbidden, map[string]string{"error": "Unauthorized path access"})
		return
	}

	info, err := os.Stat(absTargetPath)
	if err != nil || info.IsDir() {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "Audio file not found"})
		return
	}

	w.Header().Set("Content-Type", "audio/wav")
	http.ServeFile(w, r, absTargetPath)
}

func makeAssetsHandler() http.Handler {
	pythonURL, err := url.Parse(fmt.Sprintf("http://localhost:%d", pythonPort))
	if err != nil {
		log.Fatalf("[Go Backend] Failed to parse Python server URL: %v", err)
	}

	proxy := httputil.NewSingleHostReverseProxy(pythonURL)
	assetServer := application.AssetFileServerFS(assets)

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Fix for Wails 3 POST body dropping issue. Read body once for both Go handlers and proxy.
		var bodyBytes []byte
		var err error
		if r.Body != nil {
			bodyBytes, err = io.ReadAll(r.Body)
			if err == nil {
				r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
				r.ContentLength = int64(len(bodyBytes))
			} else {
				log.Printf("[Go Proxy] Error reading body: %v", err)
			}
		}

		// Specific intercepts for direct SQLite handling in Go
		switch r.URL.Path {
		case "/api/wails/select_folder":
			handleSelectFolder(w, r)
			return
		case "/api/audio":
			handleServeAudio(w, r)
			return
		case "/api/timbre/list":
			handleListTimbres(w, r)
			return
		case "/api/timbre/save":
			handleSaveTimbre(w, r)
			return
		case "/api/timbre/delete":
			handleDeleteTimbre(w, r)
			return
		case "/api/project/list":
			handleListProjects(w, r)
			return
		case "/api/project/load":
			handleLoadProject(w, r)
			return
		case "/api/project/delete":
			handleDeleteProject(w, r)
			return
		case "/api/project/reset":
			handleResetProject(w, r)
			return
		case "/api/project/state":
			handleProjectState(w, r)
			return
		}

		// Forward other /api/ calls (e.g. init, generate, update, stitch) to Python server
		if strings.HasPrefix(r.URL.Path, "/api/") {
			log.Printf("[Go Proxy] Forwarding %s %s to Python server", r.Method, r.URL.Path)
			
			// Re-populate request body from the read byte buffer
			if len(bodyBytes) > 0 {
				r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
				r.ContentLength = int64(len(bodyBytes))
			}
			
			proxy.ServeHTTP(w, r)
			return
		}
		// Serve UI static files
		assetServer.ServeHTTP(w, r)
	})
}

func Main(appAssets embed.FS) {
	assets = appAssets

	// Set up production paths if running in a bundle
	setupProductionEnvironment()

	// Initialize database on boot.
	initDB()

	// Find a free port for Python server dynamically
	if port, err := getFreePort(); err == nil {
		pythonPort = port
		log.Printf("[Go Backend] Dynamically allocated Python server port: %d", pythonPort)
	} else {
		log.Printf("[Go Backend] Warning: Could not find a free port, defaulting to %d: %v", pythonPort, err)
	}
	os.Setenv("PORT", strconv.Itoa(pythonPort))
	os.Setenv("VITE_PYTHON_PORT", strconv.Itoa(pythonPort))

	// Start Python engine
	pm := NewPythonServerManager()
	if err := pm.Start(); err != nil {
		log.Printf("[Go Backend] Warning: Could not start Python server automatically: %v. Please start it manually.", err)
	}
	defer pm.Stop()

	// Initialize the Wails desktop app
	app := application.New(application.Options{
		Name:        "Tuti",
		Description: "Tuti Studio - AI Voice Compiler App",
		Assets: application.AssetOptions{
			Handler: makeAssetsHandler(),
		},
		Mac: application.MacOptions{
			ApplicationShouldTerminateAfterLastWindowClosed: true,
		},
	})

	wailsApp = app

	// Create application window
	app.Window.NewWithOptions(application.WebviewWindowOptions{
		Title: "Tuti — AI Voice Compiler Studio",
		Width:  1280,
		Height: 850,
		StartState: application.WindowStateMaximised,
		Mac: application.MacWindow{
			InvisibleTitleBarHeight: 50,
			Backdrop:                application.MacBackdropTranslucent,
			TitleBar:                application.MacTitleBarHiddenInset,
		},
		BackgroundColour: application.NewRGB(8, 9, 15),
		URL:              "/",
	})

	log.Printf("[Go Backend] Running Wails GUI application...")
	err := app.Run()
	if err != nil {
		log.Fatal(err)
	}
}
