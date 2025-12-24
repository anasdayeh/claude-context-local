"""Language initialization for tree-sitter based chunkers.

This module handles importing and registering available languages
for the tree-sitter based code chunking system.
"""

import logging
try:
    from tree_sitter import Language
    TREE_SITTER_AVAILABLE = True
except ImportError:
    Language = None
    TREE_SITTER_AVAILABLE = False

logger = logging.getLogger(__name__)

def get_availiable_language():
    """
    Return a map {language: language_obj}
    """
    # Try to import language bindings
    res = {}

    if not TREE_SITTER_AVAILABLE:
        logger.warning("tree-sitter module not found. Code chunking will fallback to text.")
        return res

    # Prefer tree-sitter-language-pack when available (broader coverage)
    try:
        from tree_sitter_language_pack import get_language

        def _try_pack(name: str, alias: str | None = None) -> None:
            try:
                res[alias or name] = get_language(name)
            except Exception:
                logger.debug(f"tree-sitter-language-pack missing language: {name}")

        for lang in [
            "python", "javascript", "typescript", "tsx", "svelte",
            "go", "rust", "java", "c", "cpp", "csharp",
            "markdown", "html", "css", "json", "astro",
            "yaml", "toml", "xml", "graphql"
        ]:
            _try_pack(lang)

        if "javascript" in res and "jsx" not in res:
            res["jsx"] = res["javascript"]

        if "typescript" in res and "tsx" not in res:
            res["tsx"] = res["typescript"]
    except ImportError:
        logger.debug("tree-sitter-language-pack not installed")

    try:
        import tree_sitter_python as tspython
        if 'python' not in res:
            res['python'] = Language(tspython.language())
    except ImportError:
        logger.debug("tree-sitter-python not installed")

    try:
        import tree_sitter_javascript as tsjavascript
        if 'javascript' not in res:
            res['javascript'] = Language(tsjavascript.language())
        # JavaScript also supports JSX
        if 'jsx' not in res and 'javascript' in res:
            res['jsx'] = res['javascript']
    except ImportError:
        logger.debug("tree-sitter-javascript not installed for JSX")

    try:
        import tree_sitter_typescript as tstypescript
        # TypeScript has two grammars: typescript and tsx
        if 'typescript' not in res:
            res['typescript'] = Language(tstypescript.language_typescript())
        if 'tsx' not in res:
            res['tsx'] = Language(tstypescript.language_tsx())
    except ImportError:
        logger.debug("tree-sitter-typescript not installed")

    try:
        import tree_sitter_svelte as tssvelte
        if 'svelte' not in res:
            res['svelte'] = Language(tssvelte.language())
    except ImportError:
        logger.debug("tree-sitter-svelte not installed")

    try:
        import tree_sitter_go as tsgo
        if 'go' not in res:
            res['go'] = Language(tsgo.language())
    except ImportError:
        logger.debug("tree-sitter-go not installed")

    try:
        import tree_sitter_rust as tsrust
        if 'rust' not in res:
            res['rust'] = Language(tsrust.language())
    except ImportError:
        logger.debug("tree-sitter-rust not installed")

    try:
        import tree_sitter_java as tsjava
        if 'java' not in res:
            res['java'] = Language(tsjava.language())
    except ImportError:
        logger.debug("tree-sitter-java not installed")

    try:
        import tree_sitter_c as tsc
        if 'c' not in res:
            res['c'] = Language(tsc.language())
    except ImportError:
        logger.debug("tree-sitter-c not installed")

    try:
        import tree_sitter_cpp as tscpp
        if 'cpp' not in res:
            res['cpp'] = Language(tscpp.language())
    except ImportError:
        logger.debug("tree-sitter-cpp not installed")

    try:
        import tree_sitter_c_sharp as tscsharp
        if 'csharp' not in res:
            res['csharp'] = Language(tscsharp.language())
    except ImportError:
        logger.debug("tree-sitter-c-sharp not installed")

    try:
        import tree_sitter_markdown as tsmarkdown
        if 'markdown' not in res:
            res['markdown'] = Language(tsmarkdown.language())
    except ImportError:
        logger.debug("tree-sitter-markdown not installed")

    try:
        import tree_sitter_html as tshtml
        if 'html' not in res:
            res['html'] = Language(tshtml.language())
    except ImportError:
        logger.debug("tree-sitter-html not installed")

    try:
        import tree_sitter_css as tscss
        if 'css' not in res:
            res['css'] = Language(tscss.language())
    except ImportError:
        logger.debug("tree-sitter-css not installed")

    try:
        import tree_sitter_json as tsjson
        if 'json' not in res:
            res['json'] = Language(tsjson.language())
    except ImportError:
        logger.debug("tree-sitter-json not installed")

    try:
        import tree_sitter_yaml as tsyaml
        if 'yaml' not in res:
            res['yaml'] = Language(tsyaml.language())
    except ImportError:
        logger.debug("tree-sitter-yaml not installed")

    try:
        import tree_sitter_toml as tstroml
        if 'toml' not in res:
            res['toml'] = Language(tstroml.language())
    except ImportError:
        logger.debug("tree-sitter-toml not installed")

    try:
        import tree_sitter_xml as tsxml
        if 'xml' not in res:
            res['xml'] = Language(tsxml.language())
    except ImportError:
        logger.debug("tree-sitter-xml not installed")

    try:
        import tree_sitter_graphql as tsgraphql
        if 'graphql' not in res:
            res['graphql'] = Language(tsgraphql.language())
    except ImportError:
        logger.debug("tree-sitter-graphql not installed")

    return res
