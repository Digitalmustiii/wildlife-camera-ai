from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, List
from datetime import datetime, timedelta
import asyncio
import json
import base64

from database import WildlifeDatabase
from config import Config


class WildlifeAPI:
    """FastAPI application for wildlife camera"""
    
    def __init__(self, config: Config, database: WildlifeDatabase):
        self.config = config
        self.db = database
        self.app = FastAPI(title="Wildlife Camera API", version="1.0.0")
        
        # WebSocket clients
        self.ws_clients: List[WebSocket] = []
        
        # Current frame buffer (shared with detector)
        self.current_frame = None
        self.current_detections = []
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Configure CORS for frontend access"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Register API endpoints"""
        
        @self.app.get("/")
        async def root():
            return {"status": "online", "service": "Wildlife Camera API"}
        
        @self.app.get("/api/stats")
        async def get_stats():
            """Get overall statistics"""
            stats = self.db.get_total_stats()
            species_stats = self.db.get_species_stats()
            
            return {
                "total_detections": stats.get('total_detections', 0),
                "unique_species": stats.get('unique_species', 0),
                "active_days": stats.get('active_days', 0),
                "first_detection": stats.get('first_detection'),
                "last_detection": stats.get('last_detection'),
                "top_species": species_stats[:5] if species_stats else []
            }
        
        @self.app.get("/api/detections/recent")
        async def get_recent_detections(limit: int = 50, species: Optional[str] = None):
            """Get recent detections"""
            detections = self.db.get_recent_detections(limit, species)
            return {"detections": detections, "count": len(detections)}
        
        @self.app.get("/api/species")
        async def get_species_list():
            """Get all species with statistics"""
            species = self.db.get_species_stats()
            return {"species": species, "count": len(species)}
        
        @self.app.get("/api/species/{species_name}")
        async def get_species_details(species_name: str):
            """Get details for specific species"""
            detections = self.db.search_species(species_name, limit=100)
            
            if not detections:
                raise HTTPException(status_code=404, detail="Species not found")
            
            # Calculate statistics
            total = len(detections)
            avg_confidence = sum(d['confidence'] for d in detections) / total
            
            return {
                "species": species_name,
                "total_sightings": total,
                "average_confidence": avg_confidence,
                "recent_detections": detections[:10],
                "first_seen": detections[-1]['timestamp'] if detections else None,
                "last_seen": detections[0]['timestamp'] if detections else None
            }
        
        @self.app.get("/api/activity/hourly")
        async def get_hourly_activity(species: Optional[str] = None):
            """Get detection count by hour"""
            hourly = self.db.get_activity_by_hour(species)
            
            # Fill missing hours with 0
            full_hours = {h: hourly.get(h, 0) for h in range(24)}
            
            return {
                "hourly_activity": full_hours,
                "species": species,
                "total": sum(full_hours.values())
            }
        
        @self.app.get("/api/activity/daily")
        async def get_daily_summary(days: int = 7):
            """Get daily summary for last N days"""
            summary = self.db.get_daily_summary(days)
            return {"daily_summary": summary, "days": days}
        
        @self.app.get("/api/rare-sightings")
        async def get_rare_sightings(days: int = 30):
            """Get rare animal sightings"""
            rare = self.db.get_rare_sightings(days)
            return {"rare_sightings": rare, "count": len(rare)}
        
        @self.app.get("/api/search")
        async def search_detections(
            species: Optional[str] = None,
            date: Optional[str] = None,
            rarity: Optional[str] = None
        ):
            """Search detections with filters"""
            if species:
                results = self.db.search_species(species)
            elif date:
                results = self.db.get_detections_by_date(date)
            else:
                results = self.db.get_recent_detections(limit=100)
            
            # Filter by rarity if specified
            if rarity:
                results = [r for r in results if r['rarity'] == rarity]
            
            return {"results": results, "count": len(results)}
        
        @self.app.websocket("/ws/stream")
        async def websocket_stream(websocket: WebSocket):
            """WebSocket for live video stream"""
            await websocket.accept()
            self.ws_clients.append(websocket)
            
            try:
                while True:
                    # Keep connection alive
                    await asyncio.sleep(0.1)
            except WebSocketDisconnect:
                self.ws_clients.remove(websocket)
        
        @self.app.get("/api/current-frame")
        async def get_current_frame():
            """Get current frame as base64"""
            if self.current_frame is None:
                return {"frame": None, "detections": []}
            
            import cv2
            _, buffer = cv2.imencode('.jpg', self.current_frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "frame": frame_b64,
                "detections": self.current_detections,
                "timestamp": datetime.now().isoformat()
            }
    
    async def broadcast_frame(self, frame, detections):
        """Broadcast frame to all WebSocket clients"""
        if not self.ws_clients:
            return
        
        import cv2
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        message = json.dumps({
            "frame": frame_b64,
            "detections": detections,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send to all connected clients
        disconnected = []
        for client in self.ws_clients:
            try:
                await client.send_text(message)
            except:
                disconnected.append(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.ws_clients.remove(client)
    
    def update_frame(self, frame, detections):
        """Update current frame (called by detector)"""
        self.current_frame = frame
        self.current_detections = detections


def create_api(config: Config, database: WildlifeDatabase) -> FastAPI:
    """Create and configure FastAPI app"""
    api = WildlifeAPI(config, database)
    return api.app, api


# Standalone server for testing
if __name__ == "__main__":
    import uvicorn
    from config import get_config
    
    config = get_config('laptop_demo')
    db = WildlifeDatabase()
    
    app, api_instance = create_api(config, db)
    
    print("🌐 Starting Wildlife Camera API Server")
    print("📡 API docs: http://localhost:8000/docs")
    print("🔌 WebSocket: ws://localhost:8000/ws/stream")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)