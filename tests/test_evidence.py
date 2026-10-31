from tools.evidence import source_domain, independent_source_count


def test_domain_strips_www_and_subdomains():
    assert source_domain({"url": "https://www.nature.com/articles/x"}) == "nature.com"
    assert source_domain({"url": "https://news.bbc.co.uk/a"}) == "bbc.co.uk"


def test_same_domain_counts_once():
    sources = [{"url": "http://nature.com/1"}, {"url": "http://www.nature.com/2"}, {"url": "http://nature.com/3"}]
    assert independent_source_count(sources) == 1


def test_distinct_domains_counted():
    assert independent_source_count([{"url": "http://nature.com/1"}, {"url": "http://science.org/2"}]) == 2


def test_low_quality_domains_not_counted():
    sources = [{"url": "https://www.youtube.com/watch?v=abc"}, {"url": "http://nature.com/1"}]
    assert independent_source_count(sources) == 1


def test_gemini_redirect_uses_title_domain():
    sources = [
        {"url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc", "title": "nature.com"},
        {"url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/def", "title": "science.org"},
    ]
    assert independent_source_count(sources) == 2


def test_empty_and_malformed_sources():
    assert independent_source_count([]) == 0
    assert independent_source_count([{"url": ""}, {}]) == 0
