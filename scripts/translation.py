"""Translation of user-visible texts (Spanish).

Texts are written in English in the code and wrapped in ag_tr()/ag_trf(). The Qt
catalog (i18n/ArcGeekCalculator_es.qm) is tried first; if it has no entry and
QGIS runs in Spanish, the dictionary in translation_es.py is used. Any other
language, or a text without translation, shows the original English text.

Only display texts go through here: field names, identifiers and parameter keys
are never translated.
"""

from qgis.PyQt.QtCore import QCoreApplication, QSettings

CONTEXT = "ArcGeekCalculator"

_language = None


def ui_language():
    """Two-letter code of the QGIS interface language, e.g. 'es' or 'en'."""
    global _language
    if _language is None:
        locale = ""
        try:
            from qgis.core import QgsApplication
            locale = QgsApplication.locale()
        except Exception:
            locale = ""
        if not locale:
            try:
                locale = QSettings().value("locale/userLocale", "") or ""
            except Exception:
                locale = ""
        _language = str(locale)[:2].lower() or "en"
    return _language


def install_translator(plugin_dir):
    """Load i18n/ArcGeekCalculator_<lang>.qm if it exists. Returns the translator or None."""
    import os
    from qgis.PyQt.QtCore import QTranslator

    qm_path = os.path.join(plugin_dir, "i18n", f"ArcGeekCalculator_{ui_language()}.qm")
    if not os.path.exists(qm_path):
        return None
    try:
        translator = QTranslator()
        if translator.load(qm_path):
            QCoreApplication.installTranslator(translator)
            return translator
    except Exception:
        return None
    return None


def remove_translator(translator):
    if translator is not None:
        try:
            QCoreApplication.removeTranslator(translator)
        except Exception:
            return


def ag_tr(text):
    """Translated text, or the same text when there is no translation."""
    if not isinstance(text, str) or not text.strip():
        return text
    try:
        translated = QCoreApplication.translate(CONTEXT, text)
    except Exception:
        translated = text
    if translated and translated != text:
        return translated
    if ui_language() == "es":
        from .translation_es import ES
        return ES.get(text, text)
    return text


def ag_trf(template, *args):
    """Translate *template*, then fill it with str.format(*args).

    If a translation does not accept the same arguments, the English template
    is used instead, so a translation mistake can never stop a tool.
    """
    translated = ag_tr(template)
    if translated is not template:
        try:
            return translated.format(*args)
        except (IndexError, KeyError, ValueError, TypeError, AttributeError):
            pass
    return template.format(*args)
