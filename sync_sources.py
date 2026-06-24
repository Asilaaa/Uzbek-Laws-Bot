from __future__ import annotations                                                                                      
                                                                                                                           
from pathlib import Path                                                                                                
from urllib.parse import quote                                                                                          
                                                                                                                           
from common import get_connection                                                                                       
                                                                                                                           
LAW_DOCS_DIR = Path.home() / "law-docs"
MINIO_PUBLIC_BASE_URL = "http://YOUR_PUBLIC_IP:9000"
MINIO_BUCKET = "laws"
MINIO_PREFIX = "raw"                                                                                                    
                                                                                                                           
                                                                                                                           
def build_source_url(object_key: str) -> str:                                                                           
   return f"{MINIO_PUBLIC_BASE_URL}/{MINIO_BUCKET}/{quote(object_key, safe='/')}"                                      
                                                                                                                           
                                                                                                                           
def main() -> None:                                                                                                     
    files = sorted(path for path in LAW_DOCS_DIR.iterdir() if path.is_file())                                           
    if not files:                                                                                                       
       print(f"No files found in {LAW_DOCS_DIR}")                                                                      
       return                                                                                                          
                                                                                                                           
    with get_connection() as conn:                                                                                      
       with conn.cursor() as cur:                                                                                      
           for path in files:                                                                                          
               document_name = path.name                                                                               
               object_key = f"{MINIO_PREFIX}/{document_name}"                                                          
               source_url = build_source_url(object_key)                                                               
                                                                                                                           
               cur.execute(                                                                                            
                    """                                                                                                 
                   INSERT INTO law_documents (document_name, object_key, source_url)                                   
                   VALUES (%s, %s, %s)                                                                                 
                   ON CONFLICT (document_name) DO UPDATE                                                               
                   SET object_key = EXCLUDED.object_key,                                                               
                       source_url = EXCLUDED.source_url                                                                
                   """,                                                                                                
                   (document_name, object_key, source_url),                                                            
               )                                                                                                       
       conn.commit()                                                                                                   
                                                                                                                           
    print(f"Synced {len(files)} documents into law_documents")                                                          
                                                                                                                           
                                                                                                                           
if __name__ == "__main__":                                                                                              
       main()