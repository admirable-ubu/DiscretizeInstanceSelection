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
 * SortByDistance.java
 * Copyright (C) 2010 Universidad de Burgos
 */

package main.core.algorithm.sort;

import java.io.Serializable;
import java.util.Vector;

import main.core.algorithm.AlgorithmReg;
import main.core.util.InstanceIS;

import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.NearestNeighbourSearch;

/**
 * <b>Descripción</b><br>
 * Algoritmo de ordenación de instancias.
 * <p>
 * <b>Detalles</b><br>
 * Ordena las instancias en función a la distancia con su enemigo más próximo.
 * </p>
 * <p>
 * <b>Funcionalidad</b><br>
 * Cálculo de la distancia a su enemigo más próximo.
 * Ordenación de instancias en función a la distancia a su enemigo más próximo.
 * </p>
 * 
 * @author Álvar Arnáiz González
 * @version 1.2
 */
public class SortByDistance  implements Serializable {
	
	/**
	 * Para la serialización.
	 */
	private static final long serialVersionUID = -7914042695823986003L;

	/**
	 * Conjunto de instancias a ordenar.
	 */
	private Instances mToOrderSet;
	
	/**
	 * Conjunto de instancias ordenadas.
	 */
	private Instances mOrderedSet;
	
	/**
	 * Vector con las distancias de cada instancia a su enemigo más próximo.
	 */
	private double[] mDistancesToNearEnemy;
	
	/**
	 * Función distancia a emplear.
	 */
	private DistanceFunction mDistanceFunction;
	
	/**
	 * Vector de índices para identificar cada instancia dentro del conjunto a ordenar.
	 * Numerando el conjunto a ordenar desde 0 hasta número de instancias - 1 en este atributo se
	 * almacena el índice de la instancia en el conjunto a ordenar.
	 */
	private Vector<Integer> mInputDatasetIndex;
	
	/**
	 * Vector de índices para identificar cada instancia dentro del conjunto ordenado.
	 * Numerando el conjunto inicial desde 0 hasta número de instancias - 1 en este atributo se
	 * almacena el índice de la instancia en el conjunto original.
	 */
	private Vector<Integer> mOutputDatasetIndex;
	
	/**
	 * Constructor por defecto de la ordenación de instancias.
	 * Por defecto utilizará la distancia euclídea para el cálculo de distancias y fabricará el 
	 * índice desde 0 hasta número de instancias - 1.
	 * 
	 * @param toOrder Conjunto de instancias a ordenar.
	 */
	public SortByDistance (Instances toOrder) {
		this(toOrder, new EuclideanDistance(toOrder));
	} // SortByDistance
	
	/**
	 * Constructor de la ordenación de instancias.
	 * La función de distancias debe tener asignadas las instancias antes de llamar a este
	 * constructor.
	 * 
	 * @param toOrder Conjunto de instancias a ordenar.
	 * @param distanceFunction Función de distancias a utilizar.
	 */
	public SortByDistance (Instances toOrder, DistanceFunction distanceFunction) {
		this(toOrder, getIndexVector(toOrder.numInstances()), new EuclideanDistance(toOrder));
	} // SortByDistance
	
	/**
	 * Constructor de la ordenación de instancias.
	 * Por defecto utilizará la distancia euclídea para el cálculo de distancias.
	 * 
	 * @param toOrder Conjunto de instancias a ordenar.
	 * @param inputIndex Vector con los índices de las instancias a ordenar.
	 */
	public SortByDistance (Instances toOrder, Vector<Integer> inputIndex) {
		this(toOrder, inputIndex, new EuclideanDistance(toOrder));
	} // SortByDistance
	
	/**
	 * Constructor de la ordenación de instancias.
	 * La función de distancias debe tener asignadas las instancias antes de llamar a este constructor.
	 * 
	 * @param toOrder Conjunto de instancias a ordenar.
	 * @param inputIndex Vector con los índices de las instancias a ordenar.
	 * @param distanceFunction Función de distancias a utilizar.
	 */
	public SortByDistance (Instances toOrder, Vector<Integer> inputIndex, DistanceFunction distanceFunction) {
		mToOrderSet = toOrder;
		mOrderedSet = null;
		mDistancesToNearEnemy = new double[mToOrderSet.numInstances()];
		mDistanceFunction = distanceFunction;
		mInputDatasetIndex = inputIndex;
		mOutputDatasetIndex = new Vector<Integer>(mToOrderSet.numInstances());
	} // SortByDistance

	/**
	 * Devuelve el conjunto de instancias a ordenar. 
	 * 
	 * @return Conjunto de instancias a ordenar.
	 */
	public Instances getToOrderSet () {
		
		return mToOrderSet;
	} // getTrainSet
	
	/**
	 * Devuelve el conjunto ordenado de instancias.
	 * Antes de invocar este método debe haberse ejecutado el método orderByNearestEnemy, sino devolverá null. 
	 * 
	 * @return Conjunto de instancias ordenadas.
	 */
	public Instances getOrderedSet () {
		
		return mOrderedSet;
	} // getOrderedSet
	
	/**
	 * Devuelve el vector de índices del conjunto solución.
	 * 
	 * @return Vector de índices del conjunto solución seleccionado.
	 */
	public Vector<Integer> getOutputDatasetIndex () {
		
		return mOutputDatasetIndex;
	} // getOutputDatasetIndex
	
	/**
	 * Devuelve el vector con las distancias al enemigo más próximo. 
	 * Antes de invocar este método debe haberse ejecutado el método orderByNearestEnemy, sino devolverá el
	 * vector a ceros. 
	 * 
	 * @return Distancias al enemigo más próximo.
	 */
	public double[] getDistancesToNearEnemy () {
		
		return mDistancesToNearEnemy;
	} // getDistancesToNearEnemy
	
	/**
	 * Ordena el conjunto en función a la distancia al enemigo más cercano.
	 * La ordenación se realiza en función de la distancia al enemigo más próximo, el orden se establece por
	 * parámetro.
	 * 
	 * @param sortLowestToHighest Verdadero si se desea ordenar de menor a mayor, falso en caso
	 * 		  contrario.
	 * @throws Exception Excepción producida en el cálculo de distancias. 
	 */
	public void orderByNearestEnemy (boolean sortLowestToHighest) {
		// Recorrer cada instancia y calcular su distancia al enemigo más próximo.
		for (int i = 0; i < mToOrderSet.numInstances(); i++)
			mDistancesToNearEnemy[i] = getNearestEnemyDistance(mToOrderSet.instance(i),
			                                                   mToOrderSet, mDistanceFunction);
		
		// Ordenar las instancias en función a la distancia de su enemigo más cercano.
		mOrderedSet = getSortInstances(mToOrderSet, mDistancesToNearEnemy, sortLowestToHighest);
	} // orderByNearestEnemy
	
	/**
	 * Ordena el conjunto en función a la distancia al enemigo más cercano.
	 * La ordenación se realiza en función de la distancia al enemigo más próximo, el orden se establece por
	 * parámetro.
	 * 
	 * @param neighbours Vecinos más cercanos de cada una de las instancias.
	 * @param sortLowestToHighest Verdadero si se desea ordenar de menor a mayor, falso en caso
	 * 		  contrario.
	 * @param alpha Multiplicador del radio de la soft-class calculada como \alpha · std(Y(nn)).
	 * @throws Exception Excepción producida en el cálculo de distancias. 
	 */
	public void orderByNearestEnemyReg (Vector<Vector<Instance>> neighbours, boolean sortLowestToHighest, 
	                                    double alpha) {
		// Recorrer cada instancia y calcular su distancia al enemigo más próximo.
		for (int i = 0; i < mToOrderSet.numInstances(); i++)
			mDistancesToNearEnemy[i] = getNearestEnemyDistanceReg(neighbours, mToOrderSet.instance(i),
			                                                      mToOrderSet, mDistanceFunction, 
			                                                      alpha);
		
		// Ordenar las instancias en función a la distancia de su enemigo más cercano.
		mOrderedSet = getSortInstances(mToOrderSet, mDistancesToNearEnemy, sortLowestToHighest);
	} // orderByNearestEnemy
	
	/**
	 * Ordena mediante Quicksort las instancias en función del vector que se le pasa por parámetro.
	 * El criterio de ordenación se decide en función del parámetro sortLowestToHighest.
	 * A medida que actualiza las instancias, de manera simultánea, va actualizando el vector de índices de
	 * salida.
	 * 
	 * @param instancesToSort Instancias a ordenar.
	 * @param arrayToSort Vector a partir del cual se ordenarán las instancias.
	 * @param sortLowestToHighest Verdadero si se desea ordenar de menor a mayor, falso en caso
	 *        contrario.
	 * @return Instancias ordenadas.
	 */
	private Instances getSortInstances (Instances instancesToSort, double[] arrayToSort,
	                                    boolean sortLowestToHighest) {
		Instances tmpInstances;
		double indexOfInstances[] = getIndexArray(instancesToSort.numInstances());
		
		// Ordenación por QuickSort.
		NearestNeighbourSearch.quickSort(arrayToSort, indexOfInstances, 0, arrayToSort.length - 1);
		
		// Creamos un conjunto de instancias vacío.
		tmpInstances = new Instances(instancesToSort, instancesToSort.numInstances());
		
		// Almacenar en tmpInstances las instancias en el orden devuelto por quicksort.
		// Mantener al mismo tiempo el vector de índices de salida.
		if (sortLowestToHighest)
			for (int i = 0; i < indexOfInstances.length ; i++) {
				tmpInstances.add(instancesToSort.instance((int)indexOfInstances[i]));
				mOutputDatasetIndex.add(mInputDatasetIndex.get((int)indexOfInstances[i]));
			}
		else
			for (int i = indexOfInstances.length - 1; i >= 0; i--) {
				tmpInstances.add(instancesToSort.instance((int)indexOfInstances[i]));
				mOutputDatasetIndex.add(mInputDatasetIndex.get((int)indexOfInstances[i]));
			}
		
		return tmpInstances;
	} // getSortInstances
	
	/**
	 * Devuelve la distancia al enemigo más cercano.
	 * Recorre el conjunto de instancias que se le pasa y calcula con distanceFunction la instancia mas
	 * próxima cuya clase sea distinta.
	 * 
	 * @param target Instancia de la cual se va a obtener la distancia al enemigo más próximo.
	 * @param setOfInstances Conjunto donde buscar la instancia.
	 * @param distanceFunction Función con la que se calculará las distancias.
	 * @return Distancia al enemigo más próximo.
	 */
	public double getNearestEnemyDistance (Instance target, Instances setOfInstances,
	                                              DistanceFunction distanceFunction) {
		Instance insTmp;
		double dis, disToNearEnemy = Double.MAX_VALUE;
		
		// Recorrer todas las instancias de setOfInstances.
		for (int j = 0; j < setOfInstances.numInstances(); j++) {
			insTmp = setOfInstances.instance(j);
			
			// Si la instancia consultada es de otra clase calcular la distancia.
			if (target.classValue() != insTmp.classValue()) {
				// Calcular la distancia entre target e insTmp.
				dis = distanceFunction.distance(target, insTmp);
			
				// Si la distancia calculada es menor que la distancia almacenada actualizar la 
				// distancia a su enemigo. 
				if (dis < disToNearEnemy)
					disToNearEnemy = dis;
			}
		}

		return disToNearEnemy;
	} // getNearestEnemyDistance
	
	/**
	 * Devuelve la distancia al enemigo más cercano.
	 * Recorre el conjunto de instancias que se le pasa y calcula con distanceFunction la instancia mas
	 * próxima cuya clase sea distinta.
	 * 
	 * @param target Instancia de la cual se va a obtener la distancia al enemigo más próximo.
	 * @param vNeighbours Vector de vectores de vecinos más cercanos + 1. No se tendrá en cuenta el último.
	 * @param setOfInstances Conjunto donde buscar la instancia.
	 * @param distanceFunction Función con la que se calculará las distancias.
	 * @param alpha Multiplicador del radio de la soft-class calculada como \alpha · std(Y(nn)).
	 * @return Distancia al enemigo más próximo.
	 */
	public double getNearestEnemyDistanceReg (Vector<Vector<Instance>> vNeighbours, Instance target, 
	                                          Instances setOfInstances, DistanceFunction distanceFunction, 
	                                          double alpha) {
		Vector<Instance> neighbours;
		Instance insTmp;
		double theta, dis, disToNearEnemy = Double.MAX_VALUE;
		int pos;

		// Obtener la posición de la instancia a analizar.
		pos = InstanceIS.getPosOfInstance(setOfInstances, target);
		
		// Copiar la lista de vecinos sin incluir la última que es el vecino (k+1).
		neighbours = new Vector<Instance>(vNeighbours.elementAt(pos).size() - 1);

		for (int i = 0; i < vNeighbours.elementAt(pos).size() - 1; i++)
			neighbours.add(vNeighbours.elementAt(pos).get(i));

		// Calcular su theta.
		theta = AlgorithmReg.getTheta(neighbours, alpha, target.classIndex());

		// Recorrer todas las instancias de setOfInstances.
		for (int j = 0; j < setOfInstances.numInstances(); j++) {
			insTmp = setOfInstances.instance(j);
			
			// Si la instancia consultada es de otra "clase" calcular la distancia.
			if (Math.abs(target.classValue() - insTmp.classValue()) > theta) {
				// Calcular la distancia entre target e insTmp.
				dis = distanceFunction.distance(target, insTmp);
			
				// Si la distancia calculada es menor que la distancia almacenada actualizar la 
				// distancia a su enemigo. 
				if (dis < disToNearEnemy)
					disToNearEnemy = dis;
			}
		}

		return disToNearEnemy;
	} // getNearestEnemyDistance
	
	/**
	 * Ordena mediante Quicksort las instancias en función del vector que se le pasa por parámetro.
	 * 
	 * @param instancesToSort Instancias a ordenar.
	 * @param arrayToSort Vector a partir del cual se ordenarán las instancias. En este parámetro
	 *        se devuelve el índice de la ordenación.
	 * @param sortLowestToHighest Verdadero si se desea ordenar de menor a mayor, falso en caso contrario.
	 * @return Vector de instancias ordenadas.
	 */
	public static Vector<Instance> getSortVectorOfInstances (Instances instancesToSort,
	                                                         double[] arrayToSort,
	                                                         boolean sortLowestToHighest) {
		
		// Devolver las instancias ordenadas en un vector.
		return getSortVectorOfInstances(InstanceIS.getVectorOfInstance(instancesToSort),
		                                getIndexArray(instancesToSort.numInstances()), arrayToSort, 
		                                sortLowestToHighest);
	} // getSortVectorOfInstances

	/**
	 * Ordena mediante Quicksort las instancias en función del vector que se le pasa por parámetro.
	 * 
	 * @param instancesToSort Instancias a ordenar.
	 * @param arrayToSort Vector a partir del cual se ordenarán las instancias. En este parámetro se devuelve
	 *        el índice de la ordenación.
	 * @param sortLowestToHighest Verdadero si se desea ordenar de menor a mayor, falso en caso contrario.
	 * @return Vector de instancias ordenadas.
	 */
	public static Vector<Instance> getSortVectorOfInstances (Vector<Instance> instancesToSort,
	                                                         double[] arrayToSort,
	                                                         boolean sortLowestToHighest) {
		
		// Devolver las instancias ordenadas en un vector.
		return getSortVectorOfInstances(instancesToSort, getIndexArray(instancesToSort.size()),
		                                arrayToSort, sortLowestToHighest);
	} // getSortVectorOfInstances

	/**
	 * Ordena mediante Quicksort las instancias en función del vector que se le pasa por parámetro.
	 * 
	 * @param instancesToSort Instancias a ordenar.
	 * @param indexOfInstances Índice de las instancias a ordenar.
	 * @param arrayToSort Vector a partir del cual se ordenarán las instancias. 
	 * @param sortLowestToHighest Verdadero si se desea ordenar de menor a mayor, falso en caso contrario.
	 * @return Vector de instancias ordenadas.
	 */
	public static Vector<Instance> getSortVectorOfInstances (Vector<Instance> instancesToSort,
	                                                         double[] indexOfInstances,
	                                                         double[] arrayToSort,
	                                                         boolean sortLowestToHighest) {
		Vector<Instance> vectorOfInstances = new Vector<Instance>(instancesToSort.size(), 0);

		// Ordenación por QuickSort.
		NearestNeighbourSearch.quickSort(arrayToSort, indexOfInstances, 0, arrayToSort.length - 1);
		
		// Almacenar en vectorOfInstances las instancias en el orden devuelto por quicksort.
		if (sortLowestToHighest)
			for (int i = 0; i < indexOfInstances.length; i++)
				vectorOfInstances.add(instancesToSort.get((int)indexOfInstances[i]));
		else
			for (int i = indexOfInstances.length - 1; i >= 0; i--)
				vectorOfInstances.add(instancesToSort.get((int)indexOfInstances[i]));
		
		return vectorOfInstances;
	} // getSortVectorOfInstances
	
	/**
	 * Devuelve un array de índices (tipo double) de tamaño size.
	 * El primer elemento del array tiene 0, el siguiente 1 y así sucesivamente hasta size - 1.
	 *  
	 * @param size Tamaño del array.
	 * @return Array de índices.
	 */
	public static double[] getIndexArray (int size) {
		double indexOfInstances[] = new double[size];
		
		for (int i = 0; i < indexOfInstances.length; i++)
			indexOfInstances[i] = i;

		return indexOfInstances;
	} // getDoubleIndexArray
	
	/**
	 * Crea un vector de índices de tamaño size.
	 * El primer elemento del vector contiene 0, el siguiente 1 y así sucesivamente hasta size - 1.
	 * 
	 * @param size Número de elementos del vector.
	 * @return Vector de índices.
	 */
	public static Vector<Integer> getIndexVector (int size) {
		Vector<Integer> indexOfInstances = new Vector<Integer>(size);
		
		for (int i = 0; i < size; i++)
			indexOfInstances.add(i);

		return indexOfInstances;
	} // getVectorArray
	
} // SortByDistance
