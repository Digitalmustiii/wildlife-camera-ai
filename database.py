"""
Wildlife Detection Database
SQLite-based logging and querying of wildlife detections
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager


class WildlifeDatabase:
    """Manages wildlife detection history and statistics"""
    
    def __init__(self, db_path: str = "data/wildlife.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    
    @contextmanager
    def _get_connection(self):
        """Thread-safe database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_schema(self):
        """Create database tables"""
        with self._get_connection() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    species TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    rarity TEXT NOT NULL,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    snapshot_path TEXT,
                    video_clip_path TEXT,
                    session_id TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON detections(timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_species 
                ON detections(species);
                
                CREATE INDEX IF NOT EXISTS idx_rarity 
                ON detections(rarity);
                
                CREATE TABLE IF NOT EXISTS species_catalog (
                    species TEXT PRIMARY KEY,
                    first_seen DATETIME NOT NULL,
                    last_seen DATETIME NOT NULL,
                    total_sightings INTEGER DEFAULT 1,
                    rarity TEXT NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    total_detections INTEGER DEFAULT 0,
                    unique_species INTEGER DEFAULT 0
                );
            ''')
    
    def log_detection(self, detection: Dict, snapshot_path: Optional[str] = None,
                     video_path: Optional[str] = None, session_id: Optional[str] = None) -> int:
        """
        Log a detection event
        
        Args:
            detection: Detection dict from detector
            snapshot_path: Path to saved snapshot
            video_path: Path to saved video clip
            session_id: Current session identifier
            
        Returns:
            Detection ID
        """
        with self._get_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO detections 
                (timestamp, species, confidence, rarity, 
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                 snapshot_path, video_clip_path, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                detection['species'],
                detection['confidence'],
                detection['rarity'],
                *detection['bbox'],
                snapshot_path,
                video_path,
                session_id
            ))
            
            # Update species catalog
            self._update_species_catalog(conn, detection['species'], detection['rarity'])
            
            return cursor.lastrowid
    
    def _update_species_catalog(self, conn, species: str, rarity: str):
        """Update species catalog with new sighting"""
        now = datetime.now().isoformat()
        conn.execute('''
            INSERT INTO species_catalog (species, first_seen, last_seen, rarity)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(species) DO UPDATE SET
                last_seen = ?,
                total_sightings = total_sightings + 1
        ''', (species, now, now, rarity, now))
    
    def get_recent_detections(self, limit: int = 50, 
                             species: Optional[str] = None) -> List[Dict]:
        """Get most recent detections"""
        with self._get_connection() as conn:
            query = 'SELECT * FROM detections'
            params = []
            
            if species:
                query += ' WHERE species = ?'
                params.append(species)
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
    
    def get_detections_by_date(self, date: str) -> List[Dict]:
        """Get all detections for a specific date (YYYY-MM-DD)"""
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM detections
                WHERE DATE(timestamp) = ?
                ORDER BY timestamp DESC
            ''', (date,)).fetchall()
            return [dict(row) for row in rows]
    
    def get_detections_range(self, start: datetime, end: datetime) -> List[Dict]:
        """Get detections within date range"""
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM detections
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            ''', (start.isoformat(), end.isoformat())).fetchall()
            return [dict(row) for row in rows]
    
    def get_species_stats(self) -> List[Dict]:
        """Get statistics for all species"""
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT 
                    species,
                    rarity,
                    total_sightings,
                    first_seen,
                    last_seen,
                    (SELECT COUNT(*) FROM detections d 
                     WHERE d.species = sc.species 
                     AND DATE(d.timestamp) = DATE('now')) as today_count
                FROM species_catalog sc
                ORDER BY total_sightings DESC
            ''').fetchall()
            return [dict(row) for row in rows]
    
    def get_activity_by_hour(self, species: Optional[str] = None) -> Dict[int, int]:
        """Get detection count by hour of day (0-23)"""
        with self._get_connection() as conn:
            query = '''
                SELECT 
                    CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                    COUNT(*) as count
                FROM detections
            '''
            params = []
            
            if species:
                query += ' WHERE species = ?'
                params.append(species)
            
            query += ' GROUP BY hour ORDER BY hour'
            
            rows = conn.execute(query, params).fetchall()
            return {row['hour']: row['count'] for row in rows}
    
    def get_daily_summary(self, days: int = 7) -> List[Dict]:
        """Get daily detection summary for last N days"""
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as total_detections,
                    COUNT(DISTINCT species) as unique_species,
                    GROUP_CONCAT(DISTINCT species) as species_list
                FROM detections
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            ''', (days,)).fetchall()
            return [dict(row) for row in rows]
    
    def search_species(self, species: str, limit: int = 100) -> List[Dict]:
        """Search detections by species name (case-insensitive)"""
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM detections
                WHERE LOWER(species) LIKE LOWER(?)
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (f'%{species}%', limit)).fetchall()
            return [dict(row) for row in rows]
    
    def get_rare_sightings(self, days: int = 30) -> List[Dict]:
        """Get rare animal sightings from last N days"""
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM detections
                WHERE rarity = 'rare'
                AND timestamp >= datetime('now', '-' || ? || ' days')
                ORDER BY timestamp DESC
            ''', (days,)).fetchall()
            return [dict(row) for row in rows]
    
    def get_total_stats(self) -> Dict:
        """Get overall database statistics"""
        with self._get_connection() as conn:
            stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_detections,
                    COUNT(DISTINCT species) as unique_species,
                    COUNT(DISTINCT DATE(timestamp)) as active_days,
                    MIN(timestamp) as first_detection,
                    MAX(timestamp) as last_detection
                FROM detections
            ''').fetchone()
            return dict(stats) if stats else {}
    
    def cleanup_old_detections(self, days: int = 30):
        """Delete detections older than N days"""
        with self._get_connection() as conn:
            deleted = conn.execute('''
                DELETE FROM detections
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            ''', (days,)).rowcount
            return deleted
    
    def export_to_csv(self, filepath: str, species: Optional[str] = None):
        """Export detections to CSV"""
        import csv
        
        detections = self.search_species(species, limit=999999) if species else \
                     self.get_recent_detections(limit=999999)
        
        if not detections:
            return 0
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=detections[0].keys())
            writer.writeheader()
            writer.writerows(detections)
        
        return len(detections)