import urllib.request


def main():
    url = "http://127.0.0.1:8000/static/visualizations/correlation_heatmap.png"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            print("status", r.status)
            print("content-type", r.headers.get("content-type"))
            chunk = r.read(64)
            print("bytes-read", len(chunk))
    except Exception as e:
        print("error", str(e))


if __name__ == "__main__":
    main()

