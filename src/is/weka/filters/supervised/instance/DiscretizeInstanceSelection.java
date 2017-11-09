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
 * DiscretizeInstanceSelection.java
 * Copyright (C) 2013 Universidad de Burgos
 */

package weka.filters.supervised.instance;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.Enumeration;
import java.util.Vector;

import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;

import weka.filters.Filter;
import weka.filters.SupervisedFilter;
import weka.filters.unsupervised.attribute.Discretize;

/**
 * <b>Descripción</b><br>
 * Filtro que permite utilizar algoritmos de selección de instancias de clasificación en problemas
 * de regresión.
 * <p>
 * <b>Detalles</b><br>
 * Implementa los siguientes algoritmos de la superclase <code>InstanceSelection.java</code>
 * <p>
 * <b>Funcionalidad</b><br>
 * Utiliza la biblioteca de algoritmos de selección de instancias realizada para el proyecto de final
 * de carrera en la Universidad de Burgos. Tutelado por: César García Osorio y Juan José Rodríguez Díez.
 * </p>
 * 
 * @author Álvar Arnáiz González
 * @version 1.4
 */
public class DiscretizeInstanceSelection extends Filter implements SupervisedFilter, OptionHandler, InstanceSelectionFilterIF {

	/**
	 * Serial UID.
	 */
	private static final long serialVersionUID = 8190462201379437589L;

	/** 
	 * Filtro como paso previo a la selección de instancias: discretizar. 
	 */
	protected Filter m_Discretizer = new weka.filters.unsupervised.attribute.Discretize();

	/**
	 * Método de selección de instancias utilizado.
	 */
	protected Filter m_InstanceSelection = new weka.filters.supervised.instance.InstanceSelection();

	/**
	 * Permite forzar el número de clusters a la raíz cuadrada del número de instancias.
	 */
	protected boolean m_ForceBinNumber = false;

	/**
	 * Tiempo de CPU utilizado en el filtrado.
	 */
	protected long mCPUTimeElapsed;
	
	/**
	 * Tiempo utilizado en el filtrado.
	 */
	protected long mUserTimeElapsed;
	
	/**
	 * Conjunto de instancias devuelto por el algoritmo de selección de instancias tras el filtrado.
	 */
	private Instances mSolutionSet;
	
	/**
	 * Constructor por defecto.
	 * Establece el algoritmo ENN y las opciones de discretizado.
	 */
	public DiscretizeInstanceSelection () {
		super();
		
		m_Discretizer = new weka.filters.unsupervised.attribute.Discretize();
		
		String[] isOpts = new String[2];
		isOpts[0] = "-T";
		isOpts[1] = Integer.toString(InstanceSelection.TYPE_ENN);
		
		String[] filterOpts = new String[4];
		filterOpts[0] = "-unset-class-temporarily";
		filterOpts[1] = "-R";
		filterOpts[2] = "last";
		filterOpts[3] = "-O";
		
		try {
			((InstanceSelection)m_InstanceSelection).setOptions(isOpts);
			((Discretize)m_Discretizer).setOptions(filterOpts);
		} catch (Exception e) {}
	} // 
	
	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>(2);
		
		newVector.addElement(new Option("\tFull class name of filter to use, followed\n"
		                              + "\tby filter options.\n"
		                              + "\teg: \"weka.filters.unsupervised.attribute.Discretize\"",
		                                "D", 1, "-D <filter specification>"));

		newVector.addElement(new Option("\tFull class name of instance selection to use, followed\n"
		                              + "\tby filter options.\n", "S", 1, "-S <filter specification>"));


		newVector.addElement(new Option("\tForze bin number: square root of instance number.\n"
		                              + "\tCaution: only use with weka.filters.unsupervised.attribute.Discretize",
		                                "R", 0, "-R"));

		return newVector.elements();
	} // listOptions

	/**
	 * Parses a given list of options.
	 *
	 * @param options the list of options as an array of strings.
	 * @throws Exception if an option is not supported.
	 */
	public void setOptions (String[] options) throws Exception {
		// Same for filter
		String filterString = Utils.getOption('D', options);
		
		if (filterString.length() > 0) {
			String [] filterSpec = Utils.splitOptions(filterString);
			
			if (filterSpec.length == 0)
				throw new IllegalArgumentException("Invalid filter specification string");
			
			String filterName = filterSpec[0];
			filterSpec[0] = "";
			setDiscretizer ((Filter) Utils.forName(Filter.class, filterName, filterSpec));
		} else {
			String[] discOpts = new String[4];
			discOpts[0] = "-unset-class-temporarily";
			discOpts[1] = "-R";
			discOpts[2] = "last";
			discOpts[3] = "-O";

			setDiscretizer ((Filter) Utils.forName(Filter.class, defaultFilterString(), discOpts));
		}

		// Same for Instance Selection
		filterString = Utils.getOption('S', options);
		
		if (filterString.length() > 0) {
			String [] filterSpec = Utils.splitOptions(filterString);
			
			if (filterSpec.length == 0)
				throw new IllegalArgumentException("Invalid instance selection specification string");
			
			String filterName = filterSpec[0];
			filterSpec[0] = "";
			setIS((Filter) Utils.forName(Filter.class, filterName, filterSpec));
		} else {
			setIS(new weka.filters.supervised.instance.InstanceSelection());
		}

		// Si hay que forzar el número de bins como la raíz cuadrada del número de instancias
		setForceBinNumber(Utils.getFlag('R', options));
	} // setOptions

	/**
	 * Gets the current settings of the Classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String [] getOptions() {
		String [] options = new String [5];
		int current = 0;

	    if (getForceBinNumber())
	    	options[current++] = "-R";
	    else
	    	options[current++] = "";
	    
		options[current++] = "-D";
		options[current++] = "" + getFilterSpec();

		options[current++] = "-S";
		options[current++] = "" + getISSpec();
		
		return options;
	} // getOptions

	/**
	 * Sets the filter
	 *
	 * @param filter the filter with all options set.
	 */
	public void setDiscretizer (Filter filter) {

		m_Discretizer = filter;
	} // setFilter

	/**
	 * Gets the filter used.
	 *
	 * @return the filter
	 */
	public Filter getDiscretizer () {

		return m_Discretizer;
	} // getFilter

	/**
	 * Sets the instance selection filter.
	 *
	 * @param filter the filter with all options set.
	 */
	public void setIS (Filter filter) {

		m_InstanceSelection = filter;
	} // setIS

	/**
	 * Gets the instance selection filter used.
	 *
	 * @return the filter.
	 */
	public Filter getIS() {

		return m_InstanceSelection;
	} // getIS
	
	/**
	 * Devuelve si se desea forzar el número de bins a raíz cuadada del número de instancias.
	 * 
	 * @return True si se fuerza, false utilizará el número definido en el método de discretización.
	 */
	public boolean getForceBinNumber () {
		
		return m_ForceBinNumber;
	} // getForzeBinNumber
	
	/**
	 * Establece si se desea forzar el número de bins a raíz cuadada del número de instancias.
	 * 
	 * @param b True para forzar, false utilizará el número definido en el método de discretización.
	 */
	public void setForceBinNumber (boolean b) {
		m_ForceBinNumber = b;
	} // setForzeBinNumber

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return Number of bins forced to instance number square root.
	 */
	public String forceBinNumberTipText () {
		
		return "Bins number forced to instance number square root. Only works with weka.filters.unsupervised.attribute.Discretize";
	} // forzeBinNumberTipText
	
	/**
	 * Gets the filter specification string, which contains the class name of
	 * the filter and any options to the filter.
	 *
	 * @return the filter string.
	 */
	protected String getFilterSpec() {
		Filter c = getDiscretizer();
		
		if (c instanceof OptionHandler)
			return c.getClass().getName() + " " + Utils.joinOptions(((OptionHandler)c).getOptions());
		
		return c.getClass().getName();
	} // getFilterSpec

	/**
	 * Gets the instance selection filter specification string, which contains the class name of
	 * the filter and any options to the filter.
	 *
	 * @return the IS filter string.
	 */
	protected String getISSpec() {
		Filter c = getIS();
		
		if (c instanceof OptionHandler)
			return c.getClass().getName() + " " + Utils.joinOptions(((OptionHandler)c).getOptions());
		
		return c.getClass().getName();
	} // getISSpec


	/**
	 * Establece el formato de entrada de las instancias.
	 * 
	 * @param instanceInfo Una colección de instancias que contiene la estructura de entrada (cualquier
	 * instancia contenida en el conjunto será ignorada; solo se utilizará la estructura/cabecera).
	 * @return Verdadero si contiene una estructra válida.
	 * @throws Exception Si el formato de entrada no ha podido ser establecido satisfactoriamente.
	 */
	public boolean setInputFormat (Instances instanceInfo) throws Exception {
		super.setInputFormat(instanceInfo);
		super.setOutputFormat(instanceInfo);
	    
		return true;
	} // setInputFormat

	/**
	 * Introduce una nueva instancia al filtro.
	 * El filtro requiere que todas las instancias de entrenamiento sean leídas antes de producir la salida.
	 *
	 * @param instance Instancia de entrada.
	 * @return Verdadero si la instancia puede ser introducida al filtro.
	 * @throws IllegalStateException Si no se ha definido la estructura de entrada de las instancias.
	 */
	public boolean input (Instance instance) {
		if (getInputFormat() == null)
			throw new IllegalStateException("No input instance format defined");
		
		// Si es un nuevo batch.
		if (m_NewBatch) {
			resetQueue();
			m_NewBatch = false;
		}
		
		// Si se ha realizado el primer batch.
		if (m_FirstBatchDone) {
			push(instance);
			return true;
		}
		else {
			bufferInput(instance);
			return false;
		}
	} // input

	/**
	 * Establece que el filtro debe finalizar la entrada de instancias.
	 * Procesará las instancias almacenadas para devolver, mediante <code>output()</code> las instancias
	 * seleccionadas tras el filtrado.
	 *
	 * @return Verdadero si las instancias han sido procesadas.
	 * @throws IllegalStateException Si no se ha definido la estructura de las instancias.
	 * @throws Exception Si el algoritmo devuelve algun error durante la selección de instancias.
	 */
	public boolean batchFinished () throws Exception {
		// Si no se dispone de la cabecera.
		if (getInputFormat() == null)
			throw new IllegalStateException("No input instance format defined");
		
		// Realizar la selección de instancias.
		if (!m_FirstBatchDone) {
			mSolutionSet = new Instances(getInputFormat(), getInputFormat().numInstances()/10);
			filter(getInputFormat());
		}
		
		flushInput();

		m_NewBatch = true;
		m_FirstBatchDone = true;
		
		return (numPendingOutput() != 0);
	} // batchFinished
	
	/**
	 * Discretiza el conjunto de entrada, aplica sobre él la selección de instancias y deshace
	 * el paso de discretizado.
	 * 
	 * @throws Exception Si ocurre alguna excepción durante el proceso.
	 */
	public void filter (Instances inst) throws Exception {
		ThreadMXBean thMonitor = ManagementFactory.getThreadMXBean();
		boolean canMeasureCPUTime = thMonitor.isThreadCpuTimeSupported();
		Instances filteredInstances;
		
		// Fuerza el número de bins a la raíz cuadrarda del número de instancias del dataset.
		if (m_ForceBinNumber) {
			int num = (int)Math.ceil(Math.sqrt(getInputFormat().numInstances()));
			
			if (num < 2)
				num = 2;
			
			// Establecer el número de bins y forzar a utilizarlo
			((Discretize)m_Discretizer).setBins(num);
			((Discretize)m_Discretizer).setUseBinNumbers(true);
		}
		
		m_Discretizer.setInputFormat(inst);
		setInputFormat(inst);
		
		// Si se puede medir la CPU
		if(canMeasureCPUTime && !thMonitor.isThreadCpuTimeEnabled())
			thMonitor.setThreadCpuTimeEnabled(true);
		
		long thID = Thread.currentThread().getId();
		long CPUStartTime=-1, userTimeStart;
		
		userTimeStart = System.currentTimeMillis();
		
		if(canMeasureCPUTime)
			CPUStartTime = thMonitor.getThreadUserTime(thID);

		// Discretizar.
		filteredInstances = Filter.useFilter(inst, m_Discretizer);
		
		// Filtrado mediante selección de instancias sobre el conjunto discretizado.
		m_InstanceSelection.setInputFormat(filteredInstances);
		Filter.useFilter(filteredInstances, m_InstanceSelection);
		
		if(canMeasureCPUTime)
			mCPUTimeElapsed = (thMonitor.getThreadUserTime(thID) - CPUStartTime) / 1000000;
		
		mUserTimeElapsed = System.currentTimeMillis() - userTimeStart;
		
		thMonitor = null;
		
		// Deshacer la discretización.
		resetQueue();

		for (int i = 0; i < ((InstanceSelectionFilterIF)m_InstanceSelection).getSolutionSet().numInstances(); i++)
			for (int j = 0; j < inst.numInstances(); j++)
				if (equals(((InstanceSelectionFilterIF)m_InstanceSelection).getSolutionSet().instance(i), inst.instance(j), inst.classIndex())) {
					push(inst.instance(j));
					mSolutionSet.add(inst.instance(j));
					break;
				}
		
		// Si solo retiene una instancia, triplicarla para que no fallen métodos como REPTree.
		if (mSolutionSet.numInstances() == 1) {
			Instance dup;
			for (int i = 0; i < 2; i++) {
				dup = new DenseInstance(mSolutionSet.firstInstance());
				mSolutionSet.add(dup);
				dup.setDataset(inst);
				push(dup);
			}
		}
	} // filter

	/**
	 * Returns the Capabilities of this filter.
	 * 
	 * @return the capabilities of this object
	 * @see Capabilities
	 */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		
		// class
		result.disable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.DATE_CLASS);

		return result;
	} // getCapabilities
	
	/**
	 * Devuelve el conjunto de instancias devuelto por el algoritmo.
	 * 
	 * @return Conjunto de instancias solución.
	 */
	public Instances getSolutionSet() {
		
		return ((InstanceSelectionFilterIF)m_InstanceSelection).getSolutionSet();
	} // getSolutionSet
	
	/**
	 * Devuelve el tiempo de CPU empleado en el filtrado por el algoritmo de selección de instancias 
	 * seleccionado.
	 * 
	 * @return Tiempo de CPU utilizado en el filtrado.
	 */
	public long getFilterCPUTime () {
	
		return mCPUTimeElapsed;
	} // getFilterCPUTime
	
	/**
	 * Devuelve el tiempo empleado en el filtrado por el algoritmo de selección de instancias 
	 * seleccionado.
	 * 
	 * @return Tiempo utilizado en el filtrado.
	 */
	public long getFilterUserTime () {
	
		return mUserTimeElapsed;
	} // getFilterUserTime

	/**
	 * Comprueba si dos instancias son iguales en comparación de todos los atributos menos la clase.
	 * 
	 * @param i1 Instancia a comparar.
	 * @param i2 Instancia a comparar.
	 * @param classIndex Índice del atributo que se ignorará para compraobar la igualdad de las 
	 * instancias. 
	 * @return Verdadero si son iguales, falso en caso contrario.
	 */
	public static boolean equals (Instance i1, Instance i2, int classIndex) {
		// Comprobar el número de atributos.
		if (i1.numAttributes() != i2.numAttributes())
			return false;
		
		// Comprobar el peso.
		if (i1.weight() != i2.weight())
			return false;
		
		// Recorrer todos los atributos comparando los de ambas instancias.
		for (int i = 0; i < i1.numAttributes(); i++)
			if (i != classIndex) {
				// Si i1 no tiene valor para el atributo e i2 si -> no son iguales.
				if (i1.isMissing(i) && !i2.isMissing(i))
					return false;
				
				// Si i2 no tiene valor para el atributo e i1 si -> no son iguales.
				if (i2.isMissing(i) && !i1.isMissing(i))
					return false;
				
				// Si ambos atributos tienen valor y no es el mismo -> no son iguales.
				if (!i2.isMissing(i) && !i1.isMissing(i) && i1.value(i) != i2.value(i))
					return false;
			}
		
		return true;
	} // equals

	/**
	 * String describing default discretizer.
	 */
	protected String defaultFilterString () {

		return "weka.filters.unsupervised.attribute.Discretize";
	} // defaultDiscretizeString

} // DiscretizeInstanceSelection
