"""Compare shipped profile routes, support labels and model licence coverage."""
import json
import re
from pathlib import PurePosixPath

from package_audit import LICENCE_LINK_RE, MODELS, MODELS_SRC

CHECK_ID = "profile-scope"
PRE_SNAPSHOT = True
SCOPE = "win/tools/profile-scope.json"
SCHEMA = "brilliance-profile-scope/1"
PROFILE_ID = r"[\w.-]+"
GATED = re.compile(r"(?m)^# The repository is gated:")


def profile_models(text):
    """Read the block-style models map used by the shipped YAML files."""
    blocks = re.findall(r"(?ms)^models:[ \t]*(?:#[^\r\n]*)?\r?\n(.*?)(?=^[^\s#]|\Z)", text)
    if len(blocks) != 1:
        raise ValueError("expected one block-style models map")
    routes = {}
    route = None
    for line in blocks[0].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        alias = re.fullmatch(rf"  ({PROFILE_ID}):\s*(?:#.*)?", line)
        model = re.fullmatch(r"    model:[ \t]+(.+?)(?:[ \t]+#.*)?", line.rstrip())
        if alias:
            route = alias[1]
            if route in routes:
                raise ValueError(f"duplicate model route {route}")
            routes[route] = None
        elif model and route and routes[route] is None:
            file = model[1].strip().strip("\"'").replace(chr(92), "/")
            routes[route] = "/".join(PurePosixPath(file).parts[-3:])
        elif not line.startswith("    ") or model or line.startswith("    model:"):
            raise ValueError(f"unexpected models map line {line.strip()!r}")
    if not routes or None in routes.values():
        raise ValueError("each model route needs a model: file")
    return routes


def sections(markdown):
    parts = re.split(r"(?m)^## ([^\r\n]+)\r?\n", markdown)
    return dict(zip(parts[1::2], parts[2::2]))


def table_rows(markdown):
    rows = {}
    for line in markdown.splitlines():
        if line.startswith("|"):
            cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line)[1:-1]]
            if cells:
                rows[cells[0]] = cells[1:]
    return rows


def support_labels(markdown):
    labels = {}
    matrix = sections(markdown).get("Models and card counts", "")
    for cell in re.split(r"(?<!\\)\||\n", matrix):
        label = re.match(r"\s*\*\*([^*]+)\*\*", cell)
        profiles = re.search(rf"\bProfiles?:\s*(`{PROFILE_ID}`(?:,\s*`{PROFILE_ID}`)*)", cell)
        if label and profiles:
            for name in re.findall(r"`([^`]+)`", profiles[1]):
                if name in labels:
                    raise ValueError(f"SUPPORT.md lists {name} twice in Models and card counts")
                labels[name] = label[1]
    return labels


def plain(text):
    return " ".join(text.split())


def verify_terms(model, licence_text, table, parts, errors):
    name = model.get("terms_row", model["index_row"])
    row = table.get(name, [])
    terms = model.get("terms")
    restricted = bool(model["use_restrictions"])
    if len(row) != 2 or (row[1] != "None") != restricted or bool(terms) != restricted:
        errors.append(f"TERMS-OF-USE.md: {name} and profile-scope.json disagree on use restrictions")
    if not terms:
        return
    heading = terms["heading"]
    section = plain(parts.get(heading, ""))
    if not section:
        errors.append(f"TERMS-OF-USE.md: missing pass-through section {heading!r}")
        return
    markers = ["/terms/licenses/" + model["licence"], *terms["required_text"]]
    for marker in markers:
        if plain(marker) not in section:
            errors.append(f"TERMS-OF-USE.md: {heading!r} is missing {marker!r}")
    licence = plain(licence_text)
    for start, end in terms.get("quoted_licence_sections", []):
        first = licence.find(start)
        last = licence.find(end, first + len(start)) if end else len(licence)
        if first < 0 or last < 0 or licence[first:last].strip() not in section:
            errors.append(f"TERMS-OF-USE.md: {heading!r} must carry the full licence passage starting {start!r}")


def run(ctx):
    if ctx.zip and not ctx.source:
        return {"verdict": "skip", "reason": "profile-scope.json needs the candidate source clone (--source)"}
    try:
        scope = json.loads(ctx.source_read(SCOPE))
        if scope.get("schema") != SCHEMA:
            raise ValueError(f"expected schema {SCHEMA}")
        profiles, models = scope["profiles"], scope["models"]
    except (FileNotFoundError, KeyError, ValueError) as e:
        return {"verdict": "fail", "reason": f"{SCOPE}: {e}"}

    names = set(ctx.zip_names() if ctx.zip else ctx.source_names())
    read = ctx.zip_read if ctx.zip else ctx.source_read
    prefix = "" if ctx.zip else "win/"
    model_dir = MODELS if ctx.zip else MODELS_SRC
    errors = []

    def read_text(path):
        if path in names:
            content = read(path).decode("utf-8-sig")
            if content.strip():
                return content
        errors.append(f"{path} is missing or empty")
        return ""

    profile_dir = prefix + "profiles/"
    files = {}
    for path in map(PurePosixPath, sorted(names)):
        if path.parent.as_posix() + "/" == profile_dir and path.suffix in (".yaml", ".yml"):
            if path.stem in files:
                errors.append(f"{path}: duplicate shipped profile name {path.stem}")
            files[path.stem] = path.as_posix()
    labels = support_labels(read_text(prefix + "SUPPORT.md"))
    for kind, actual in (("shipped profiles", files), ("SUPPORT.md labels", labels)):
        for name in sorted(set(actual) - profiles.keys()):
            errors.append(f"{name}: in {kind}, missing from {SCOPE}")
        for name in sorted(profiles.keys() - set(actual)):
            errors.append(f"{name}: in {SCOPE}, missing from {kind}")
    used = set()
    for name, profile in profiles.items():
        if name in labels and labels[name] != profile["support"]:
            errors.append(f"{name}: support label {profile['support']!r} differs from SUPPORT.md ({labels[name]!r})")
        expected = {}
        for route, model_id in profile["models"].items():
            if model_id not in models:
                errors.append(f"{name}: {route} names unknown model {model_id!r}")
                continue
            used.add(model_id)
            expected[route] = models[model_id]["file"]
        if name not in files:
            continue
        text = read_text(files[name])
        try:
            actual = profile_models(text)
        except ValueError as e:
            errors.append(f"{files[name]}: {e}")
            continue
        if expected != actual:
            errors.append(f"{files[name]}: expected model routes {expected}, found {actual}")
        # The generated Sandbox inherits each model's gate from its source profile.
        if name != "sandbox" and expected and len(expected) == len(profile["models"]):
            gated = any(models[model_id]["gated"] for model_id in profile["models"].values())
            if gated != bool(GATED.search(text)):
                errors.append(f"{files[name]}: repository gating disagrees with {SCOPE}")

    index = table_rows(read_text(model_dir + "/index.md"))
    terms = sections(read_text(prefix + "TERMS-OF-USE.md"))
    terms_table = table_rows(terms.get("The models and their licences", ""))
    for model_id, model in models.items():
        if model_id not in used:
            errors.append(f"{model_id}: listed in {SCOPE} but used by no profile")
        if type(model["gated"]) is not bool:
            errors.append(f"{model_id}: gated must be true or false")
        links = set(LICENCE_LINK_RE.findall(" ".join(index.get(model["index_row"], []))))
        licence_text = ""
        for licence_id in [model["licence"], *model.get("notices", [])]:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", licence_id):
                errors.append(f"{model_id}: invalid licence ID {licence_id!r}")
                continue
            if licence_id not in links:
                errors.append(f"{model_dir}/index.md: {model['index_row']} must link /terms/licenses/{licence_id}")
            text = read_text(f"{model_dir}/{licence_id}.txt")
            if licence_id == model["licence"]:
                licence_text = text
        verify_terms(model, licence_text, terms_table, terms, errors)

    if errors:
        return {"verdict": "fail", "reason": f"{len(errors)} profile scope issue(s)", "details": errors}
    return {"verdict": "pass",
            "reason": f"{len(profiles)} profiles and {len(models)} model containers match their support and licences"}
