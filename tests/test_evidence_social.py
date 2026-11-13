from tools.evidence import independent_domains, independent_source_count


def test_social_posts_do_not_count_as_independent_sources():
    sources = [{"url": "https://briefing.tfie.io/p/x", "title": "t"},
               {"url": "https://www.linkedin.com/posts/someone-activity-123", "title": "t"},
               {"url": "https://www.reddit.com/r/batteries/comments/abc", "title": "t"}]
    assert independent_domains(sources) == {"tfie.io"}
    assert independent_source_count(sources) == 1
