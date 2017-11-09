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
 * AlgorithmReg.java
 * Copyright (C) 2014 Universidad de Burgos
 */

package main.core.algorithm;

import java.util.Vector;

import main.core.exception.NotEnoughInstancesException;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

/**
 * <b>Descripción</b><br>
 * Superclase para los algoritmos de Selección de Instancias orientados para regresión.
 * <p>
 * <b>Detalles</b><br>
 * Posibilita la adición de nuevos algoritmos con una interfaz común.
 * Posibilita el seguimiento de las instancias seleccionadas a partir del índice sobre el dataset
 * original. 
 * </p>
 * <p>
 * <b>Funcionalidad</b><br>
 * Implementa estrategia en el patrón de diseño Strategy.
 * </p>
 * 
 * @author Álvar Arnaiz González
 * @version 1.5
 */
public abstract class AlgorithmReg extends Algorithm {

	/**
	 * Para la serialización.
	 */
	private static final long serialVersionUID = 3629084375898232180L;

	/**
	 * Valor de la clase en el conjunto de datos. Necesario para el método <code>classValueOf</code>.
	 */
	protected int mClassIndex;
	
	/**
	 * Constructor por defecto del algoritmo de selección de instancias para regresión.
	 */
	public AlgorithmReg () {
		super();
		
	} // AlgorithmReg
	
	/**
	 * Constructor de los algoritmos orientados a regresión al cual se le pasa el nuevo conjunto de
	 * instancias a tratar.
	 * 
	 * @param train Conjunto de instancias a seleccionar.
	 * @throws NotEnoughInstancesException Si se ha producido algún error en la clasificación.
	 */
	public AlgorithmReg (Instances train) throws NotEnoughInstancesException {
		super(train);
		
		// Almacenar el valor de la clase.
		mClassIndex = train.classIndex();
	} // AlgorithmReg
	
	/**
	 * Constructor de los algoritmos orientados a regresión al cual se le pasa el nuevo conjunto de
	 * instancias a tratar.
	 * 
	 * @param train Conjunto de instancias a seleccionar.
	 * @param inputDatasetIndex Array de índices para identificar cada instancia dentro del conjunto
	 * inicial de instancias.
	 * @throws NotEnoughInstancesException Si se ha producido algún error en la clasificación.
	 */
	public AlgorithmReg (Instances train, int [] inputDatasetIndex) throws NotEnoughInstancesException {
		super(train, inputDatasetIndex);

		// Almacenar el valor de la clase.
		mClassIndex = train.classIndex();
	} // AlgorithmReg

	/**
	 * Devuelve el umbral theta calculado como &Theta; = &alpha; · std(Y(X<sub>s</sub>)).
	 * 
	 * @param subset Subconjunto de instancias a utilizar.
	 * @param alpha Coeficiente a utilizar en el cálculo.
	 * @return Valor umbral en regresión, 0 en cualquier otro caso.
	 */
	public static double getTheta (Instances subset, double alpha) {
		
		return getTheta(subset, alpha, subset.firstInstance().classIndex());
	} // getTheta
	
	/**
	 * Devuelve el umbral theta calculado como &Theta; = &alpha; · std(Y(X<sub>s</sub>)).
	 * Se utiliza para calcular el umbral a partir de una serie de instancias (la clase debe ser
	 * numérica) y un valor alfa. 
	 * 
	 * @param subset Subconjunto de instancias a utilizar.
	 * @param alpha Coeficiente a utilizar en el cálculo.
	 * @param classIndex Índice del atributo que contiene la clase. 
	 * @return Valor umbral en regresión, 0 en cualquier otro caso.
	 */
	public static double getTheta (Instances subset, double alpha, int classIndex) {
		Vector<Instance> vSubset = new Vector<Instance>(subset.numInstances());
		
		for (int i = 0; i < subset.numInstances(); i++)
			vSubset.add(subset.instance(i));
		
		return getTheta(vSubset, alpha, classIndex);
	} // getTheta
	
	/**
	 * Devuelve el umbral theta calculado como &Theta; = &alpha; · std(Y(X<sub>s</sub>)).
	 * Se utiliza para calcular el umbral a partir de una serie de instancias (la clase debe ser
	 * numérica) y un valor alfa. 
	 * 
	 * @param subset Subconjunto de instancias a utilizar.
	 * @param alpha Coeficiente a utilizar en el cálculo.
	 * @param classIndex Índice del atributo que contiene la clase. 
	 * @return Valor umbral en regresión, 0 en cualquier otro caso.
	 */
	public static double getTheta (Vector<Instance> subset, double alpha, int classIndex) {
		double sigma = 0.0, mean = 0.0;
		
		// Si solo existe una instancia (o ninguna) devuelve alfa.
		if (subset.size() <= 1)
			return alpha;

		// Si el valor de la clase no es numérico devolver 0.
		if (!subset.firstElement().attribute(classIndex).isNumeric())
			return 0.0;
		
		// Calcular la media
		for (Instance inst : subset)
			mean += inst.value(classIndex);
		
		mean /= subset.size();
		
		// Calcular la desviación típica
		for (Instance inst : subset)
			sigma += Math.pow(inst.value(classIndex) - mean, 2);
		
		sigma = Math.sqrt(sigma / (subset.size() - 1));
		
		return sigma * alpha;
	} // getTheta
	
	/**
	 * Devuelve el valor de la clase para la instancia dada.
	 * No se tiene en cuenta el valor de la clase de la instancia sino la que tiene almacenada el 
	 * algoritmo.
	 * 
	 * @param inst Instancia de la que se desea obtener la clase.
	 * @return Clase de la instancia.
	 */
	public double classValueOf (Instance inst) {
		
		return inst.value(mClassIndex);
	} // classValueOf

	/**
	 * Se entrena un kNN con el conjunto que se le pasa y si la diferencia entre el valor devuelto por el
	 * predictor y el valor de la instancia es mayor al umbral &Theta; es que no se clasifica correctamente.
	 * Utiliza todas las instancias del conjunto trainSet para el cálculo del vecino más cercano, es decir,
	 * <code>k = trainSet.numInstances()</code> 
	 * 
	 * @param instance Instancia a clasificar.
	 * @param trainSet Conjunto de vecinos de la instancia a evaluar.
	 * @param theta Valor umbral por el que se considera una instancia igual o distinta a otra.
	 * @return Verdadero si no se clasifica correctamente por sus vecinos, falso en caso contrario.
	 * @throws Exception Excepción en el cálculo de vecinos cercanos.
	 */
	public static boolean isMisclassified(Instance instance, Instances trainSet, double theta) throws Exception {

		return isMisclassified(instance, trainSet, theta, trainSet.numInstances());
	} // isMisclassified

	/**
	 * Se entrena un kNN con el conjunto que se le pasa y si la diferencia entre el valor devuelto por el
	 * predictor y el valor de la instancia es mayor al umbral &Theta; es que no se clasifica correctamente. 
	 * 
	 * @param instance Instancia a clasificar.
	 * @param trainSet Conjunto de vecinos de la instancia a evaluar.
	 * @param theta Valor umbral por el que se considera una instancia igual o distinta a otra.
	 * @param numNN Número de vecinos cercanos a utilizar en el cálculo.
	 * @return Verdadero si no se clasifica correctamente por sus vecinos, falso en caso contrario.
	 * @throws Exception Excepción en el cálculo de vecinos cercanos.
	 */
	public static boolean isMisclassified(Instance instance, Instances trainSet, double theta, int numNN) throws Exception {
		IBk ibk = new IBk(numNN);

		// 20141201 -> Probar con la opción de que tenga en cuenta la distancia de los vecinos para asignar la clase.
//		String[] options = new String[1];
//		options[0] = "-I";
//		ibk.setOptions(options);
		// 20141203 -> No funciona mejor

		// Construir el clasificador.
		ibk.buildClassifier(trainSet);
		
		if (Math.abs(ibk.classifyInstance(instance) - instance.classValue()) > theta)
			return true;
		
		return false;
	} // isMisclassified

} // AlgorithmReg
