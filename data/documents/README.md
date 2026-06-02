- Yêu cầu : Crawl tất cả dữ liệu từ HTML pages , PDF documents , ... từ các nguồn liên quan đến Đại học VNU và các trường thành viên
- Mô tả :
  - Giới thiệu chung và Lịch sử của VNU (nguồn https://vnu.edu.vn và trang wikipedia của VNU)
  - Thông tin tuyển sinh : Bao gồm thông tin về các chương trình học, quy trình tuyển sinh và yêu cầu đầu vào của VNU nói chung và tất cả các trường thành viên của VNU (Nguồn vnu.edu.vn, uet.vnu.edu.vn, ussh.vnu.edu.vn...  ) ( nhớ là tất cả các trường nhé)
  - Quy định học thuật : Các văn bản quy phạm về giáo dục: Bao gồm các quyết định và hướng dẫn liên quan đến quy định đào tạo đại học và sau đại học (vidu các quy định đào tạo thạc sĩ, tiến sĩ, cao học, đại học, ...)
  - Chương trình học thuật : Chương trình đào tạo chung của từng trường từng ngành, đào tạo tiến sĩ , thạc sĩ, cao học, Thông tin về các chương trình đào tạo liên kết với đối tác quốc tế và các chương trình giảng dạy bằng tiếng Anh.
  - Tổng hợp các nguồn chính :
    - vnu.edu.vn
    - wikipidia của VNU
    - uet.vnu.edu.vn
    - ussh.vnu.edu.vn
    - (Các trường thành viên khác của VNU )
    - Cả text HTML và file PDF

* Định dạng yêu cầu file json (trong data có một số file mẫu có sẵn của một số trường, ko cần crawl lại những trường đó):
  ```json
  {
    "_id": {...},
    "url": "...",
    "title": "...",
    "content": "...",
    "domain": "...",
    "category": [...],
    "create_at": {...}
  }
  ```
