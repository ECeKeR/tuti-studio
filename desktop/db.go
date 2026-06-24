package desktop

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

var db *sql.DB

type Timbre struct {
	Name         string `json:"name"`
	VoiceMode    string `json:"voice_mode"`
	Speaker      string `json:"speaker"`
	DesignPrompt string `json:"design_prompt"`
	RefAudio     string `json:"ref_audio"`
	RefText      string `json:"ref_text"`
	ModelSize    string `json:"model_size"`
	CreatedAt    string `json:"created_at"`
	SFTEnabled   bool   `json:"sft_enabled"`
}

type Project struct {
	ID         int    `json:"id"`
	Name       string `json:"name"`
	ScriptText string `json:"script_text"`
	StateJSON  string `json:"state_json"`
	CreatedAt  string `json:"created_at"`
	UpdatedAt  string `json:"updated_at"`
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, in)
	if err != nil {
		return err
	}
	return out.Sync()
}

func initDB() {
	cwd, err := os.Getwd()
	if err != nil {
		log.Fatalf("[DB] Failed to get cwd: %v", err)
	}

	dbDir := filepath.Join(cwd, "pipeline_work")
	err = os.MkdirAll(dbDir, 0755)
	if err != nil {
		log.Fatalf("[DB] Failed to create database directory: %v", err)
	}

	dbPath := filepath.Join(dbDir, "tuti.db")
	oldDbPath := filepath.Join(dbDir, "voiceforge.db")

	// Migrate old database to new name if needed
	if _, err := os.Stat(dbPath); os.IsNotExist(err) {
		if _, errOld := os.Stat(oldDbPath); errOld == nil {
			log.Printf("[DB] Migrating old database from %s to %s...", oldDbPath, dbPath)
			errMove := os.Rename(oldDbPath, dbPath)
			if errMove != nil {
				log.Printf("[DB] Failed to rename old database: %v. Trying to copy...", errMove)
				errCopy := copyFile(oldDbPath, dbPath)
				if errCopy != nil {
					log.Printf("[DB] Failed to copy old database: %v. Starting with new empty database.", errCopy)
				} else {
					log.Printf("[DB] Database migration successful (copied).")
				}
			} else {
				log.Printf("[DB] Database migration successful (renamed).")
			}
		}
	}

	log.Printf("[DB] Opening SQLite database at %s...", dbPath)

	dbConn, err := sql.Open("sqlite", dbPath)
	if err != nil {
		log.Fatalf("[DB] Failed to open SQLite database: %v", err)
	}

	db = dbConn
	db.SetMaxOpenConns(1)

	if _, err := db.Exec("PRAGMA journal_mode=WAL;"); err != nil {
		log.Printf("[DB] Failed to set journal_mode=WAL: %v", err)
	}
	if _, err := db.Exec("PRAGMA synchronous=NORMAL;"); err != nil {
		log.Printf("[DB] Failed to set synchronous=NORMAL: %v", err)
	}
	if _, err := db.Exec("PRAGMA busy_timeout=30000;"); err != nil {
		log.Printf("[DB] Failed to set busy_timeout: %v", err)
	}

	// Create tables
	queries := []string{
		`CREATE TABLE IF NOT EXISTS timbres (
			name TEXT PRIMARY KEY,
			voice_mode TEXT NOT NULL,
			speaker TEXT,
			design_prompt TEXT,
			ref_audio TEXT,
			ref_text TEXT,
			model_size TEXT DEFAULT '1.7B',
			created_at TEXT NOT NULL
		)`,
		`CREATE TABLE IF NOT EXISTS projects (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			name TEXT NOT NULL,
			script_text TEXT,
			state_json TEXT NOT NULL,
			created_at TEXT NOT NULL,
			updated_at TEXT NOT NULL
		)`,
		`CREATE TABLE IF NOT EXISTS settings (
			key TEXT PRIMARY KEY,
			value TEXT
		)`,
	}

	for _, q := range queries {
		_, err := db.Exec(q)
		if err != nil {
			log.Fatalf("[DB] Failed to execute schema query: %v", err)
		}
	}
	log.Printf("[DB] Database schema initialized successfully.")
	resetStuckGeneratingSegments()
}

func resetStuckGeneratingSegments() {
	if db == nil {
		return
	}
	log.Println("[DB] Checking for stuck 'generating' segments on boot...")
	rows, err := db.Query("SELECT id, state_json FROM projects")
	if err != nil {
		log.Printf("[DB] Error querying projects for stuck segments: %v", err)
		return
	}
	defer rows.Close()

	type projectUpdate struct {
		id        int
		stateJSON string
	}
	var updates []projectUpdate

	for rows.Next() {
		var id int
		var stateJSON string
		if err := rows.Scan(&id, &stateJSON); err != nil {
			log.Printf("[DB] Error scanning project state: %v", err)
			continue
		}

		if !strings.Contains(stateJSON, "\"generating\"") {
			continue
		}

		// Parse the JSON state
		var state map[string]interface{}
		if err := json.Unmarshal([]byte(stateJSON), &state); err != nil {
			log.Printf("[DB] Error unmarshaling state JSON for project %d: %v", id, err)
			continue
		}

		segmentsVal, ok := state["segments"]
		if !ok {
			continue
		}

		segmentsSlice, ok := segmentsVal.([]interface{})
		if !ok {
			continue
		}

		updated := false
		for _, segVal := range segmentsSlice {
			segMap, ok := segVal.(map[string]interface{})
			if !ok {
				continue
			}
			if segMap["status"] == "generating" {
				log.Printf("[DB] Resetting stuck segment in project ID %d to 'pending'", id)
				segMap["status"] = "pending"
				segMap["error_msg"] = "Generation interrupted (server restart/crash)"
				updated = true
			}
		}

		if updated {
			newStateJSON, err := json.Marshal(state)
			if err != nil {
				log.Printf("[DB] Error marshaling updated state JSON for project %d: %v", id, err)
				continue
			}
			updates = append(updates, projectUpdate{id: id, stateJSON: string(newStateJSON)})
		}
	}
	rows.Close() // Close rows to release the database connection

	// Perform updates on the connection
	for _, up := range updates {
		now := time.Now().Format(time.RFC3339)
		_, err = db.Exec("UPDATE projects SET state_json = ?, updated_at = ? WHERE id = ?", up.stateJSON, now, up.id)
		if err != nil {
			log.Printf("[DB] Error updating project %d state in DB: %v", up.id, err)
		}
	}

	log.Println("[DB] Stuck segment cleanup completed in Go.")
}

func saveTimbre(t Timbre) error {
	now := time.Now().Format(time.RFC3339)
	_, err := db.Exec(`
		INSERT OR REPLACE INTO timbres (name, voice_mode, speaker, design_prompt, ref_audio, ref_text, model_size, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`, t.Name, t.VoiceMode, t.Speaker, t.DesignPrompt, t.RefAudio, t.RefText, t.ModelSize, now)
	return err
}

func listTimbres() ([]Timbre, error) {
	rows, err := db.Query(`SELECT name, voice_mode, speaker, design_prompt, ref_audio, ref_text, model_size, created_at FROM timbres ORDER BY name ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var list []Timbre
	for rows.Next() {
		var t Timbre
		var speaker, designPrompt, refAudio, refText, modelSize sql.NullString
		err := rows.Scan(&t.Name, &t.VoiceMode, &speaker, &designPrompt, &refAudio, &refText, &modelSize, &t.CreatedAt)
		if err != nil {
			return nil, err
		}
		t.Speaker = speaker.String
		t.DesignPrompt = designPrompt.String
		t.RefAudio = refAudio.String
		t.RefText = refText.String
		t.ModelSize = modelSize.String

		// Check if SFT adapter exists
		adapterPath := filepath.Join("pipeline_work", fmt.Sprintf("%s_adapter.safetensors", t.Name))
		if _, err := os.Stat(adapterPath); err == nil {
			t.SFTEnabled = true
		} else {
			t.SFTEnabled = false
		}

		list = append(list, t)
	}
	return list, nil
}

func deleteTimbre(name string) error {
	_, err := db.Exec(`DELETE FROM timbres WHERE name = ?`, name)
	return err
}

func listProjects() ([]Project, error) {
	rows, err := db.Query(`SELECT id, name, script_text, created_at, updated_at FROM projects ORDER BY updated_at DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var list []Project
	for rows.Next() {
		var p Project
		var scriptText sql.NullString
		err := rows.Scan(&p.ID, &p.Name, &scriptText, &p.CreatedAt, &p.UpdatedAt)
		if err != nil {
			return nil, err
		}
		p.ScriptText = scriptText.String
		list = append(list, p)
	}
	return list, nil
}

func getProject(id int) (string, error) {
	var stateJSON string
	err := db.QueryRow(`SELECT state_json FROM projects WHERE id = ?`, id).Scan(&stateJSON)
	if err != nil {
		return "", err
	}
	return stateJSON, nil
}

func deleteProject(id int) error {
	_, err := db.Exec(`DELETE FROM projects WHERE id = ?`, id)
	return err
}

func getActiveProjectID() (int, error) {
	var val string
	err := db.QueryRow("SELECT value FROM settings WHERE key = 'active_project_id'").Scan(&val)
	if err != nil {
		if err == sql.ErrNoRows {
			return 0, nil
		}
		return 0, err
	}
	var id int
	_, err = fmt.Sscanf(val, "%d", &id)
	return id, err
}

func setActiveProjectID(id int) error {
	if id > 0 {
		val := fmt.Sprintf("%d", id)
		_, err := db.Exec("INSERT OR REPLACE INTO settings (key, value) VALUES ('active_project_id', ?)", val)
		return err
	} else {
		_, err := db.Exec("DELETE FROM settings WHERE key = 'active_project_id'")
		return err
	}
}

func getMostRecentProjectState() (string, error) {
	var stateJSON string
	var id int
	err := db.QueryRow("SELECT id, state_json FROM projects ORDER BY updated_at DESC LIMIT 1").Scan(&id, &stateJSON)
	if err != nil {
		if err == sql.ErrNoRows {
			return "", nil
		}
		return "", err
	}
	// Set it as active
	_ = setActiveProjectID(id)
	return stateJSON, nil
}
