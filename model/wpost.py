from flask_sqlalchemy import SQLAlchemy
import pandas as pd

db = SQLAlchemy()

class WPPosts(db.Model):
    __tablename__ = 'wp_posts'
    id = db.Column(db.Integer, primary_key=True)
    post_content = db.Column(db.Text)
    post_title = db.Column(db.String(255))



def GetToolsInfo():
    results = WPPosts.query.filter(
        WPPosts.post_content.like('%工具%'),
        WPPosts.post_title != '每日AI工具'
    ).all()
    data = [{
            # 'id': post.id,
            'post_title': post.post_title,
            'post_content': post.post_content
        } for post in results]
    df = pd.DataFrame(data)
    csv_file_path = 'results.csv'
    df.to_csv(csv_file_path, index=False, encoding='utf-8')
    print(f"Data saved to {csv_file_path}")
    
    return results