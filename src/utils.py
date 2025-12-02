import os

def use_cache(func):
    """
    Un décorateur qui vérifie si les résultats ont déjà été calculés
    en lisant un fichier texte avec des marqueurs de début (-B) et fin (-E).
    """
    filename = "graphe_cache.txt"

    def wrapper(*args, **kwargs):
        marker_start = f"{func.__name__}-B"
        marker_end = f"{func.__name__}-E"
        
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            cache_found = False
            cached_data = []
            recording = False 

            for line in lines:
                clean_line = line.strip() # Enlever les \n
                
                if clean_line == marker_start:
                    recording = True
                    cache_found = True
                    continue # On passe à la ligne suivante sans l'ajouter

                if clean_line == marker_end:
                    recording = False
                    break 

                if recording:
                    cached_data.append(clean_line)

            if cache_found and cached_data:
                print(f"Cache trouvé pour {func.__name__}")
                return cached_data

        print(f"Calcul en cours pour {func.__name__}...")
        results = func(*args, **kwargs)
        
        # On ouvre en mode 'a' (append) pour ajouter à la fin sans écraser le reste
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"{marker_start}\n")
            for r in results:
                f.write(f"{str(r)}\n")
            f.write(f"{marker_end}\n")
            
        return results

    return wrapper