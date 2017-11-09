# DiscretizeInstanceSelection: Instance Selection by means of discretization for Regression

This is an open-source filters for Weka that implements instance selection algorithms for regression.

The original Weka software is necessary: https://www.cs.waikato.ac.nz/ml/weka/

* Disc-ENN: Instance Selection for regression by means of discretization.
* Reg-ENN: Kordos, M., & Blachnik, M. (2012). Instance selection with neural networks for regression problems. Artificial Neural Networks and Machine Learning–ICANN 2012, 263-270.
* MI: Guillen, A., Herrera, L. J., Rubio, G., Pomares, H., Lendasse, A., & Rojas, I. (2010). New method for instance or prototype selection using mutual information in time series prediction. Neurocomputing, 73(10), 2030-2038.


### Cite this software as:
 **Á. Arnaiz-González, J-F. Díez Pastor, Juan J. Rodríguez, C. García Osorio.** _Instance selection for regression by discretization._ Expert Systems with Applications, 54, 340-350. [doi: 10.1016/j.eswa.2015.12.046](doi: 10.1016/j.eswa.2015.12.046)

```
@article{ArnaizGonzalez2016,
  title = "Instance selection for regression by discretization",
  journal = "Expert Systems with Applications",
  volume = "54",
  number = "Supplement C",
  pages = "340 - 350",
  year = "2016",
  issn = "0957-4174",
  doi = "10.1016/j.eswa.2015.12.046",
  url = "http://www.sciencedirect.com/science/article/pii/S095741741600049X",
  author = "\'Alvar Arnaiz-Gonz\'alez and Jos\'e F. D\'iez-Pastor and Juan J. Rodr\'iguez and C\'esar Garc\'ia-Osorio"   
}
```


# How to use

## Download and build with ant
- Download source code: It is host on GitHub. To get the sources and compile them we will need git instructions. The specifically command is:
```git clone https://github.com/alvarag/DiscretizeInstanceSelection.git ```
- Build jar file: 
```ant dist_all ```
It generates the jar file under /dist/weka



## How to run

Include the file instanceselection.jar into the path. Example: 

```java -cp instanceselection.jar:weka.jar weka.gui.GUIChooser```

The new filters can be found in: weka/filters/supervised/instance.

* DiscretizeInstanceSelection: Instance selection method (ENN) for regression by means of discretization.
* InstanceSelectionForRegression: Instance selection for regression using Mutual Information (Guillen et al. 2010) and RegENN (Kordos & Blachnik, 2012).

