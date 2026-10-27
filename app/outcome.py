"""
Turns a read result into what the wall should do and say. These are the same messages the wall
app showed (NoteScanner.ShowOutcome); Unity will only display them.
Moderation (kind words) stays on the wall, where it is shared with the tablet.
"""


def fit(text, limit):
    """Shortens text to `limit` characters, cutting at a word boundary when possible."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return (text[:cut] if cut > limit // 2 else text[:limit]).rstrip(" ,.;")


def evaluate(result, limits, english_only=False):
    name, message = result.get("name", ""), result.get("message", "")
    out = {"code": "", "kind": "warning", "title": "", "sub": "Then place it under the camera again",
           "name": "", "message": "", "language": result.get("language", "other")}

    if not result.get("ok"):
        out.update(code="reader_not_ready", title="The note reader is not ready yet",
                   sub="Please try again in a moment")
    elif not name and not message:   # the model's "readable" flag alone is not trusted
        out.update(code="unreadable", title="We couldn't read your note",
                   sub="Please write a little bigger and clearer, then place it again")
    elif not name:
        out.update(code="missing_name", title="Please write your name first",
                   sub="Add your name to the note and place it again")
    elif not message:
        out.update(code="missing_message", title="Please write your good deed",
                   sub="Add what you will do on the note and place it again")
    elif out["language"] == "other" or (english_only and out["language"] != "en"):
        out.update(code="unsupported_language",
                   title="For now, please write in English" if english_only else "Please write in English, Hindi or Gujarati")
    else:
        name = fit(name, limits["name_max"])
        out.update(code="accepted", kind="success", name=name, message=fit(message, limits["message_max"]),
                   title="Thank you, %s!" % name, sub="Your leaf is on its way to the tree")
    return out
