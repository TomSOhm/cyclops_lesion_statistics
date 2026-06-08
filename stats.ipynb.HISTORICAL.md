# stats.ipynb  Note de préservation historique

## Statut : ARCHIVÉ  ne pas modifier

Le notebook `stats.ipynb` constitue la **phase 1 exploratoire** du projet Linos Stats. Il contient l'analyse initiale **uniquement sur la cohorte Ménisque** (n=19, 38 lignes), réalisée avant la décision d'intégrer la cohorte Cyclops (n=50) pour une étude cas-témoin formelle.

## Contenu du notebook (52 cellules)

- Cellules 0-5 : chargement Excel, renommage colonnes, ProfileReport, marker `# Nice plots`
- Cellule 6 : imports (avec `%matplotlib inline`)
- Cellules 7-12 : setup (hygiène données, variables dérivées, palettes)
- Cellules 13-19 : §1 Profil démographique & morphologique
- Cellules 20-31 : §2 Associations facteurs de risque
- Cellules 32-42 : §3 Progression lésionnelle Chirurgie 1→Chirurgie 2 (analyse primaire phase 1)
- Cellules 43-50 : §4 Tendances temporelles
- Cellule 51 : Disclaimer markdown

## Pourquoi archivé

La phase 2 du projet (ce repo, après ce fichier) :
- Intègre la **cohorte Cyclops** (n=50) en plus de Ménisque (n=19)
- Refactor le code en **package Python modulaire** (`src/lino_stats/`)
- Sépare l'analyse en **7 notebooks chapitre** dans `notebooks/`
- Construit un **wiki Obsidian** structuré PhD-thesis au lieu d'un notebook monolithique

L'analyse phase 1 reste valide pour la cohorte Ménisque isolée, et constitue un point de départ pédagogique. Elle est **référencée mais non re-utilisée** dans le nouveau pipeline.

## Référence vault

Voir `Obsidian Vault/linos_stats/99-Project-Log/` pour la trace complète du projet.

## Règle

**Ne pas modifier `stats.ipynb`.** Tout travail nouveau va dans `notebooks/*.ipynb` (phase 2).
