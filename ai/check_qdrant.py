"""Check Qdrant collection status directly."""

from app.core.config import settings
from app.vector.qdrant_client import get_client

client = get_client()

try:
    # Get collection info
    collection_info = client.get_collection(settings.collection_name)
    
    print(f"Collection: {settings.collection_name}")
    print(f"Points count: {collection_info.points_count}")
    print(f"Vector size: {collection_info.config.params.vectors.size}")
    print(f"Status: {collection_info.status}")
    print()
    
    # Try to scroll through some points
    if collection_info.points_count > 0:
        print("Sample points:")
        scroll_result = client.scroll(
            collection_name=settings.collection_name,
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        for point in scroll_result[0]:
            print(f"  - ID: {point.id}")
            print(f"    URL: {point.payload.get('url', 'N/A')}")
            print(f"    Text preview: {point.payload.get('text', '')[:100]}...")
            print()
    else:
        print("⚠️  No points found in collection!")
        print()
        print("Possible reasons:")
        print("1. Ingestion is still running")
        print("2. All pages were skipped (insufficient text)")
        print("3. Ingestion encountered errors")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("Make sure:")
    print("1. Qdrant Cloud credentials are correct in .env")
    print("2. Collection was created successfully")
