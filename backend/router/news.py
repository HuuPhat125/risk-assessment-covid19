from fastapi import APIRouter, HTTPException
from utils.rss_reader import RSSReader

router = APIRouter()
rss_reader = RSSReader(keywords=["covid", "corona", "sars-cov-2", "hậu covid", "covid-19"])
@router.get("/news")
async def get_news(source: str = 'all'):
    # Kiểm tra source hợp lệ nếu không phải 'all'
    if source != 'all' and source not in rss_reader.feeds:
        raise HTTPException(status_code=400, detail=f"Source '{source}' not supported")

    news = rss_reader.get_news(source)
    return {"status": "success", "data": news}
