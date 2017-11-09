/*
 * This file is part of Instance Selection Library.
 * 
 * Instance Selection Library is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Instance Selection Library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Instance Selection Library.  If not, see <http://www.gnu.org/licenses/>.
 * 
 * MIAlgorithm.java
 * Copyright (C) 2013 Universidad de Burgos
 */

package main.core.algorithm;

import java.io.Serializable;
import java.util.Vector;

import main.core.exception.NotEnoughInstancesException;
import main.core.util.InstanceIS;
import main.core.util.LinearISNNSearch;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.neighboursearch.NearestNeighbourSearch;

/**
 * <b>Descripción</b><br>
 * Algoritmo de selección de instancias basado en Mutual Information.
 * <p>
 * <b>Detalles</b><br>
 * Calcula la información mutua.
 * Posibilita seguir al algoritmo paso a paso.
 * Códgo generado a partir de la idea propuesta por A. Guillen et al., New method for instance or 
 * prototype selection using mutual information in time series prediction, Neurocomputing (2010),
 * Volume 73, Pages 2030-2038.
 * </p>
 * <p>
 * <b>Pseudocódigo del MI prototype selection</b><br>
 * <span style="font-weight: bold;">Require:</span>
 * Training set&nbsp;<span style="font-style: italic;"></span><span style="font-style: italic;">T</span> = 
 * {(x<sub>1</sub>, y<sub>1</sub>),...,(x<sub>n</sub>, y<sub>n</sub>)}, a selector <span style="font-style:
 * italic;">S = </span><span style="font-style: italic;" class="mw-headline" id=".E2.88.85.7B.7D">&#8709;<br>
 * </span><span style="font-weight: bold;">Ensure: </span>The set of selected instances 
 * <span style="font-style: italic;">S&nbsp;&#8834;X</span><br><br>
 * 
 * <span style="font-weight: bold;"></span>
 * <span style="font-style: italic;"></span>
 * 
 * &nbsp; 1: Calculate the <span style="font-style: italic;">K</span> nearest neighbors in the input space of <span style="font-weight: bold;">x<sub>i</sub></span> = (x<sub>1</sub>, x<sub>2</sub>,...,x<sub>id</sub>) for <span style="font-style: italic;">i</span> = 1...n  (nn [<span style="font-weight: bold;">x<sub>i</sub></span>, j] for j = 1...K)<br>
 * &nbsp; 2: <span style="font-weight: bold;">for</span> i = 1...n<br>
 * &nbsp; 3: &nbsp;&nbsp;&nbsp; Calculate the mutual information value l(X,Y)<sub>i</sub> when removing
 * <span style="font-weight: bold;">x<sub>i</sub></span> from X<br>
 * &nbsp; 4: <span style="font-weight: bold;">end</span><br>
 * &nbsp; 5: Normalize l(X,Y)<sub>i</sub> in [0, 1]<br>
 * &nbsp; 6: <span style="font-weight: bold;">for</span> i = 1...n<br>
 * &nbsp; 7: &nbsp;&nbsp;&nbsp; <span style="font-style: italic;">Cdiff</span> = 0<br>
 * &nbsp; 8: &nbsp;&nbsp;&nbsp; <span style="font-weight: bold;">for</span> j = 1...
 * <span style="font-style: italic;">K</span><br>
 * &nbsp; 9: &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; <span style="font-style: italic;">diff</span> = 
 * <span style="font-style: italic;">I</span>(X,Y)<sub>nn[x<sub>i</sub>, j]</sub> - 
 * <span style="font-style: italic;">I</span>(X,Y)<sub>i</sub><br>
 *       10: &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; <span style="font-weight: bold;">if</span>
 *       <span style="font-style: italic;">diff</span> &gt &alpha;<br>
 *       11: &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; <span style="font-style: italic;">
 *       Cdiff</span> = <span style="font-style: italic;">Cdiff</span> + 1<br>
 *       12: &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; <span style="font-weight: bold;">end</span><br>
 *       13: &nbsp;&nbsp;&nbsp; <span style="font-weight: bold;">end</span><br>
 *       14: &nbsp;&nbsp;&nbsp; <span style="font-weight: bold;">if</span> <span style="font-style: italic;">
 *       Cdiff</span> &#8805 <span style="font-style: italic;">K</span><br>
 *       15: &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; Select prototype<br>
 *       16: &nbsp;&nbsp;&nbsp; <span style="font-weight: bold;">else</span><br>
 *       17: &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; Discard prototype<br>
 *       18: &nbsp;&nbsp;&nbsp; <span style="font-weight: bold;">end</span><br>
 *       19: <span style="font-weight: bold;">end</span><br>
 * </p>
 * <p>
 * <b>Funcionalidad</b><br>
 * Implementa el algoritmo Mutual Information Prototype Selection.
 * </p>
 * 
 * @author Álvar Arnaiz González
 * @version 1.4
 */
public class MIAlgorithm extends AlgorithmReg {
	
	/**
	 * Para la serialización.
	 */
	private static final long serialVersionUID = -3547152819548929013L;
	
	/**
	 * Índices a los vecinos más cercanos. 
	 */
	protected int[][] mNearestNeighbour;
	
	/**
	 * Información mutua de cada una de las instancias.
	 */
	protected double[] mMI;
	
	/**
	 * Número de pasos que el algoritmo lleva ejecutados. 
	 */
	protected int mNumOfIterations;
	
	/**
	 * Indicador de si se han calculado los vecinos cercanos. 
	 */
	protected boolean mCalcNeighbours;
	
	/**
	 * Indicador de si se ha calculado la información mutua. 
	 */
	protected boolean mCalcMI;
	
	/**
	 * Número de vecinos cercanos a buscar.
	 */
	protected int mNumOfNearestNeighbour;
	
	/**
	 * Valor de alfa: sensitividad/especificidad.
	 */
	protected double mAlpha;
	
	/**
	 * Función con los valores digamma.
	 */
	protected DigammaFunction mDigamma;
	
	/**
	 * Constructor por defecto del algoritmo MI.
	 */
	public MIAlgorithm () {
		super();
		
		// Por defecto el número de vecinos cercanos es 6.
		mNumOfNearestNeighbour = 6;
		
		// Por defecto el alfa utilizado es 0.05.
		mAlpha = 0.05;
	} // MIAlgorithm
	
	/**
	 * Constructor del algoritmo MI al que se le pasa el nuevo conjunto de instancias a tratar.
	 * Es equivalente a:<br>
	 * <code>
	 * Algorithm();<br>
	 * reset(train);
	 * </code>
	 * 
	 * @param train Conjunto de instancias a seleccionar.
	 * @throws NotEnoughInstancesException Si el dataset no tiene ninguna instancia que al eliminarla mejore
	 * su capacidad de clasificación.
	 */
	public MIAlgorithm (Instances train) throws NotEnoughInstancesException {
		super(train);
		
	} // MIAlgorithm
	
	/**
	 * Constructor del algoritmo MI al que se le pasa el nuevo conjunto de instancias a tratar.
	 * Es equivalente a:<br>
	 * <code>
	 * Algorithm(train);<br>
	 * reset(inputDatasetIndex);
	 * </code>
	 * 
	 * @param train Conjunto de instancias a seleccionar.
	 * @param inputDatasetIndex Array de índices para identificar cada instancia dentro del conjunto
	 * inicial de instancias.
	 * @throws NotEnoughInstancesException Si el dataset no tiene ninguna instancia que al eliminarla mejore
	 * su capacidad de clasificación.
	 */
	public MIAlgorithm (Instances train, int [] inputDatasetIndex) throws NotEnoughInstancesException {
		super(train, inputDatasetIndex);

	} // MIAlgorithm

	/**
	 * Devuelve el número de vecinos cercanos que va a utilizar el algoritmo. 
	 * 
	 * @return Número de vecinos cercanos a utilizar por el algoritmo.
	 */
	public int getNumOfNearestNeighbour () {
		
		return mNumOfNearestNeighbour;
	} // getNumOfNearestNeighbour
	
	/**
	 * Devuelve el valor de alfa.
	 * 
	 * @return Valor de alfa.
	 */
	public double getAlpha () {
		
		return mAlpha;
	} // getAlpha
	
	/**
	 * Establece el número de vecinos cercanos que va a utilizar el algoritmo. 
	 * 
	 * @param nn Número de vecinos cercanos a utilizar por el algoritmo.
	 * @throws IllegalArgumentException Es lanzada si el número de vecinos es menor que 1.
	 */
	public void setNumOfNearestNeighbour (int nn) {
		if (nn < 1)
			throw new IllegalArgumentException("El número de vecinos cercanos debe ser mayor de 0.");
		
		mNumOfNearestNeighbour = nn;
	} // setNumOfNearestNeighbour
	
	/**
	 * Establece el valor de alfa.
	 * Alfa debe estar en el intervalo [0, 1] y ponderla la sensitividad o la especificidad del
	 * algoritmo.
	 * 
	 * @param alpha Valor de alfa.
	 */
	public void setAlpha (double alpha) {
		if (alpha < 0 || alpha > 1)
			throw new IllegalArgumentException("El valor de alfa debe estar en el intervalo [0, 1]");
		
		mAlpha = alpha;
	} // setAlpha
	
	/**
	 * Ejecuta un paso del algoritmo.
	 * Comienza con el conjunto inicial.
	 * Calcula los vecinos más cercanos en el espacio de entrada.
	 * Calcula la información mutua en el espacio (X, Y) y la normaliza.
	 * Para cada una de las instancias se realiza un cálculo de la información mutua de sus vecinos,
	 * en función de eso y el valor alfa decide si mantiene o elimina la instancia. 
	 * 
	 * @return Verdadero si quedan pasos que ejecutar, falso en caso contratio.
	 * @throws Exception Excepción producida durante el paso del algoritmo.
	 */
	public boolean step () throws Exception {
		double diff, cDiff = 0;
		
		if (!mCalcNeighbours) {
			mCalcNeighbours = true;
			
			// Calcular los vecinos cercanos.
			calcNearestNeighbours();
		} else if (!mCalcMI){
			mCalcMI = true;
			
			// Calcular la información mutua
			calcMutualInformation();
		} else {
			// Aumentar el contador de número de iteraciones.
			mNumOfIterations++;
			
			// Recorrer cada vecino más cercano. 
			for (int j = 0; j < mNumOfNearestNeighbour; j++) {
				diff = mMI[mNearestNeighbour[mCurrInstancePos][j]] - mMI[mCurrInstancePos];
// 20131001 - Tal y como aparece en el paper no funciona, lo he vuelto a comprobar
//				diff = mMI[mCurrInstancePos] - mMI[mNearestNeighbour[mCurrInstancePos][j]]; 
				
				if (diff > mAlpha)
					cDiff++;
			}
			
			// Si la diferencia es mayor o igual que k -> se elimina la instancia.
			if (cDiff >= mNumOfNearestNeighbour) {
// 20131001 - Tal y como aparece en el paper no funciona, lo he vuelto a comprobar
//			if (cDiff < mNumOfNearestNeighbour) {
				int solutionSetPosition;
				
				// Obtener la posición de la instancia actual en el conjunto solución.
				solutionSetPosition = InstanceIS.getPosOfInstance(mSolutionSet, mCurrentInstance);
				
				// Eliminar la instancia del conjunto solución.
				mSolutionSet.delete(solutionSetPosition);
				
				// Eliminar el índice de la instancia borrada.
				mOutputDatasetIndex.remove(solutionSetPosition);
			}
			
			// Si hemos recorrido todas las instancias del dataset de entrada finalizar el algoritmo.
			if (mTrainSet.numInstances() == mNumOfIterations)
				return false;

			// Aumentar la instancia actual.
			mCurrInstancePos++;
			
			// Asignar la siguiente instancia a analizar.
			mCurrentInstance = mTrainSet.instance(mCurrInstancePos);
		}
		
		return true;
	} // step
	
	/**
	 * Reinicia el algoritmo con un conjunto de entrenamiento nuevo.
	 * Inicializa las variables de trabajo del algoritmo.
	 *  
	 * @param train Conjunto de instancias a seleccionar.
	 * @throws NotEnoughInstancesException Si el dataset no tiene ninguna instancia que al eliminarla mejore
	 * su capacidad de clasificación.
	 */
	public void reset (Instances train) throws NotEnoughInstancesException {
		super.reset(train);
	} // reset
	
	/**
	 * Reinicia el algoritmo con un conjunto de entrenamiento nuevo.
	 * Inicializa las variables de trabajo del algoritmo.
	 *  
	 * @param train Conjunto de instancias a seleccionar.
	 * @param inputDatasetIndex Array de índices para identificar cada instancia dentro del conjunto
	 * inicial de instancias.
	 * @throws NotEnoughInstancesException Si el dataset no tiene ninguna instancia que al eliminarla mejore
	 * su capacidad de clasificación.
	 */
	public void reset (Instances train, int[] inputDatasetIndex) throws NotEnoughInstancesException {
		super.reset(train, inputDatasetIndex);
		
		init();
	} // reset

	/**
	 * Inicializa las variables del algoritmo.
	 * Comienza copiando el conjunto inicial como solución.
	 */
	protected void init () throws NotEnoughInstancesException {
		// Por defecto el número de vecinos cercanos es 6.
		mNumOfNearestNeighbour = 6;
		
		// Por defecto el alfa utilizado es 0.05.
		mAlpha = 0.05;
		
		// Copiar el conjunto de entrenamiento en el conjunto solución.
		mSolutionSet = new Instances(mTrainSet);
		
		// Inicializar el vector de índices de salida.
		for (Integer index : mInputDatasetIndex)
			mOutputDatasetIndex.add(new Integer(index));
		
		// Vector con la información mutua.
		mMI = new double[mSolutionSet.numInstances()];
		
		// Inicializar el algoritmo de vecinos cercanos.
		try {
			mNearestNeighbourSearch = new LinearISNNSearch(mSolutionSet);
			mNearestNeighbourSearch.setDistanceFunction(new ManhattanDistance(mSolutionSet));
		} catch (Exception e) {
			// No debe saltar esta excepción.
			throw new NotEnoughInstancesException(NotEnoughInstancesException.MESSAGE);
		}
		
		// Inicializar los contadores y variables auxiliares.
		mCalcNeighbours = false;
		mCalcMI = false;
		mNumOfIterations = 0;
		mCurrInstancePos = 0;
		mCurrentInstance = mTrainSet.firstInstance();
		
		// Precalcular los N valores de la función digamma.
		mDigamma = new DigammaFunction(mTrainSet.numInstances());
	} // init
	
	/**
	 * Calcula los vecinos más cercanos de todas las instancias.
	 * 
	 * @throws Exception Excepción en el cálculo de vecinos cercanos.
	 */
	protected void calcNearestNeighbours () throws Exception {
		Instances kNearest;
		
		// Índices de los vecinos cercanos.
		mNearestNeighbour = new int[mSolutionSet.numInstances()][mNumOfNearestNeighbour];

		// Recorrer todas las instancias.
		for (int i = 0; i < mSolutionSet.numInstances(); i++) {
			 kNearest = mNearestNeighbourSearch.kNearestNeighbours(mSolutionSet.instance(i), mNumOfNearestNeighbour);
			 
			 // Para cada vecino cercano almacenar su índice.
			 for (int j = 0; j < mNumOfNearestNeighbour; j++)
				 mNearestNeighbour[i][j] = InstanceIS.getPosOfInstance(mSolutionSet, kNearest.instance(j));
		}
	} // calcNearestNeighbours
	
	/**
	 * Calcula la información mutua para cada una de las instancias en el espacio (X, Y).
	 * 
	 * @throws Exception en el cálculo de vecinos cercanos y/o distancias.
	 */
	protected void calcMutualInformation () throws Exception {
		Instances[] instances;
		Instances kNN;
		double distkNN;
		int nnPos, count, numAttr = mSolutionSet.numAttributes(),
		           numInst = mSolutionSet.numInstances();
		double sum, inic, maxEps;
		int classIndex = mSolutionSet.classIndex();
		
		// Generar el espacio Z = {X, Y}.
		mSolutionSet.setClassIndex(-1);
		mNearestNeighbourSearch.setInstances(mSolutionSet);
		mNearestNeighbourSearch.getDistanceFunction().setInstances(mSolutionSet);
		
		// Primera parte de la fórmula MI precalculada.
		inic = mDigamma.getDigammaValue(mNumOfNearestNeighbour) - ((numAttr - 1) / mNumOfNearestNeighbour) +
		         ((numAttr - 1) * mDigamma.getDigammaValue(numInst));
		
		// Array de datasets.
		instances = getOneInstancesPerAttribute();
		
		// Array de búsquedas lineales, una por cada atributo.
		NearestNeighbourSearch[] nnSearchPerAttr = new LinearISNNSearch[numAttr];
		
		// Generar las funciones distancia para cada uno de los atributos.
		for (int i = 0; i < nnSearchPerAttr.length; i++)
			nnSearchPerAttr[i] = new LinearISNNSearch(instances[i]);
		
		// Para cada instancia.
		for (int i = 0; i < numInst; i++) {
			sum = 0;
			maxEps = 0;
			
			// Por cada atributo calcular la distancia a sus vecinos más cercanos.
/*			for (int j = 0; j < nnSearchPerAttr.length; j++) {
				nnSearchPerAttr[j].kNearestNeighbours(instances[j].instance(i), mNumOfNearestNeighbour);
				distkNN = nnSearchPerAttr[j].getDistances();
				
				// Buscar la mayor distancia de sus k vecinos.
				for (int k = 0; k < distkNN.length; k++)
					if (distkNN[k] > maxEps)
						maxEps = distkNN[k];
			}
*/
			// Calcular los k vecinos más cercanos en el espacio Z={X,Y}.
			kNN = mNearestNeighbourSearch.kNearestNeighbours(mSolutionSet.instance(i), mNumOfNearestNeighbour);
			
			// Recorrer cada vecino más cercano para calcular la mayor norma.
			for (int j = 0; j < kNN.numInstances(); j++) {
				// Calcular la posición del vecino más cercano.
				nnPos = InstanceIS.getPosOfInstance(mSolutionSet, kNN.instance(j));
				
				// Recorrer todos los espacios buscando la mayor distancia.
				for (int k = 0; k < nnSearchPerAttr.length; k++) {
					distkNN = nnSearchPerAttr[k].getDistanceFunction().distance(instances[k].instance(i), 
					                                                            instances[k].instance(nnPos));
					
					if (distkNN > maxEps)
						maxEps = distkNN;
				}
			}

			
			// Calcular el sumatorio de la fórmula.
			for (int j = 0; j < nnSearchPerAttr.length; j++) {
				count = 0;
				
				// Contar los puntos en el espacio j más próximos que maxEps.
				for (int k = 0; k < numInst; k++)
					// No contar a la propia instancia.
					if (i != k && nnSearchPerAttr[j].getDistanceFunction().distance(
						                instances[j].instance(i), instances[j].instance(k)) < maxEps)
						count++;
				
				// Almacenamos el sumatorio de los FuncionDigamma(n_x(i))
				sum += mDigamma.getDigammaValue(count);
			}
			
			// Asignar la MI de cada instancia.
			mMI[i] = inic - (sum / numInst);
		}
		
		// Normalizar.
		normalizeMI();
		
		// Establecer la clase del conjunto de datos.
		mSolutionSet.setClassIndex(classIndex);
		
		// Liberar la memoria de la búsqueda de vecinos cercanos.
		mNearestNeighbourSearch = null;
	} // calcMutualInformation
	
	/**
	 * Devuelve un array de datasets cada uno de los cuales tiene el mismo número de instancias
	 * que el conjunto de datos original.
	 * Cada dataset devuelto contiene un atributo de tal modo que todos ellos combinados serían
	 * el conjunto original.  
	 * 
	 * @return Array de dataset cada uno de los cuales tiene un atributo.
	 */
	private Instances[] getOneInstancesPerAttribute (){
		Instances[] instances;
		Instance tmpO, tmpD;
		int numAttr = mSolutionSet.numAttributes(),
		    numInst = mSolutionSet.numInstances();
		
		// Array de datasets.
		instances = new Instances[numAttr];
		
		// Crear un array de datasets, cada uno contendrá un único atributo.
		for (int i = 0; i < instances.length; i++) {
			// Crear el dataset vacío sin clase.
			instances[i] = new Instances(mSolutionSet, numInst);
			instances[i].setClassIndex(-1);
			
			// Eliminar todos los atributos menos uno.
			for (int k = 0, j = 0; j < numAttr; j++)
				// Si es el atributo que queremos mantener no se borra.
				if (i == j)
					k++;
				else
					instances[i].deleteAttributeAt(k);
		}
		
		// Rellenar los datasets creados.
		for (int i = 0; i < numInst; i++) {
			tmpO = mSolutionSet.instance(i);
			
			// Por cada instancia genera tantas como atributos existan.
			for (int j = 0; j < instances.length; j++) {
				tmpD = new DenseInstance(1);
				tmpD.setValue(0, tmpO.value(j));
				instances[j].add(tmpD);
			}
		}
		
		return instances;
	} // getOneInstancesPerAttribute

	/**
	 * Normaliza el vector con los valores de Mutual Information.
	 */
	private void normalizeMI () {
		double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
		
		// Buscar el menor y el mayor valor.
		for (int i = 0; i < mMI.length; i++) {
			if (mMI[i] > max)
				max = mMI[i];
			
			if (mMI[i] < min)
				min = mMI[i];
		}
		
		// Normalizar.
		for (int i = 0; i < mMI.length; i++)
			mMI[i] = (mMI[i] - min) / (max - min);
	} // normalizeMI

	/**
	 * <b>Descripción</b><br>
	 * Implementa la función digamma (&#968).
	 * <p>
	 * <b>Detalles</b><br>
	 * Consiste en una función con memoria para evitar el recálculo de la función.  
	 * </p>
	 * <p>
	 * <b>Funcionalidad</b><br>
	 * Implementa la función digamma,
	 * {@link http://en.wikipedia.org/wiki/Digamma_function}.
	 * </p>
	 * 
	 * @author Álvar Arnáiz González
	 * @version 1.4
	 */
	public class DigammaFunction implements Serializable {
		
		/**
		 * Para la serialización.
		 */
		private static final long serialVersionUID = -7540804147650544995L;

		/**
		 * Constante de Euler-Mascheroni.
		 */
		private static final double EULER_MASCHERONI = -0.577215664901532;
		
		/**
		 * Valores digamma calculados.
		 */
		private Vector<Double> mDigammaValues;
		
		/**
		 * Constructor por defecto.
		 * Genera el vector de valores vacío.
		 */
		public DigammaFunction () {
			mDigammaValues = new Vector<Double>();
		} // DigammaFunction
		
		/**
		 * Constructor por defecto.
		 * Genera los k primeros valores de la función digamma.
		 * 
		 * @param k Precalcula los k primeros valores de la función digamma.
		 */
		public DigammaFunction (int k) {
			mDigammaValues = new Vector<Double>(k);
			
			calcNextValues(k);
		} // DigammaFunction
		
		/**
		 * Devuelve el valor de la función digamma para el valor k, es decir, &#968(k).
		 * El valor 0 no está definido por lo que se devuelve 0.
		 * 
		 * @param k Valor de la función digamma a calcular.
		 * @return Valor de la función digamma.
		 */
		public double getDigammaValue (int k) {
			// El valor para 0 no está definido.
			if (k == 0)
				return 0;
			
			// Si no tenemos el valor almacenado -> Calcularlo.
			if (mDigammaValues.size() < k)
				calcNextValues(k);
			
			return mDigammaValues.elementAt(k - 1);
		} // getDigammaValue
		
		/**
		 * Calcula los valores de la función digamma hasta el valor k.
		 * Almacena los valores para mejorar el rendimeinto.
		 * 
		 * @param k Valor de la función digamma hasta la que se desea calcular.
		 */
		private void calcNextValues (int k) {
			for (int i = mDigammaValues.size(); i < k; i++)
				if (i == 0)
					mDigammaValues.add(EULER_MASCHERONI);
				else
					mDigammaValues.add(mDigammaValues.elementAt(i - 1) + (1.0 / i));
		} // calcNextValues
		
	} // DigammaFunction

} // MIAlgorithm
