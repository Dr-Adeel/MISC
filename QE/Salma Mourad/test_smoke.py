def test_simple_check():
    """Vérifie que 1 est bien égal à 1 (Test de survie)"""
    assert 1 == 1

def test_import_main():
    """Vérifie si on peut au moins importer le fichier de Tarik"""
    try:
        import main
        assert True
    except ImportError:
        assert False, "Le fichier main.py est introuvable ou contient des erreurs critiques"