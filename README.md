# Clustering & HAC

This project currently supports running Hierarchical Agglomerative Clustering (HAC) with single linkage and Manhattan distance. Reports are generated in the `report` folder.

## Installation
1. Clone the repository.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script:
```bash
python main.py --report
```

## Contributing
- Please create a PR with the attached output of the file.

## Supported Scripts
### Decision Tree
Finished, need to be restructured to run easily
- [x] working

### Clustering
#### k-means
- [ ] K-means
#### HAC
  - [x] Single linkage
  - [ ] Complete linkage
  - [ ] Average linkage
   (to do)

##### distance
- [x] Manhatten Distance

### Association Rule Mining
- [ ] Apriori (to be implemented)
- [ ] FP-Growth (to be implemented)

