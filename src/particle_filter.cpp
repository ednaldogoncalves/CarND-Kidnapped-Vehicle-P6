/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// This function takes as input a GPS position, an initial heading estimate, 
// and an array of uncertainties for these measurements
void ParticleFilter::init(double x, double y, double theta, double std[])
{
	/**
	* TODO: Set the number of particles. Initialize all particles to 
	*   first position (based on estimates of x, y, theta and their uncertainties
	*   from GPS) and all weights to 1. 
	* TODO: Add random Gaussian noise to each particle.
	* NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	*/
	
	// create normal distributions for x, y, and theta
	
	// normal (Gaussian) distribution for x
	std::normal_distribution<double> dist_x(x, std[0]);
	
	// normal distributions for y and theta
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);
	
	// declare a random engine to be used across multiple
	std::default_random_engine gen;

	// Initializing the number of particles
	num_particles = 100;

	// resize the vectors of particles and weights
	particles.resize(num_particles);

	// Generate particles with normal distribution with mean on GPS values.
	for(auto& p: particles)
	{
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;
	}

	is_initialized = true;
}

// This function takes as input the amount of time between time steps,
// the velocity and yaw rate measurement uncertainties, and the current 
// time step velocity and yaw rate measurements.
// Update each particle's position estimates and account for sensor noise by adding Gaussian noise.
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) 
{
	/**
	* TODO: Add measurements to each particle and add random Gaussian noise.
	* NOTE: When adding noise you may find std::normal_distribution 
	*   and std::default_random_engine useful.
	*  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	*  http://www.cplusplus.com/reference/random/default_random_engine/
	*/

	// declare a random engine to be used across multiple
	std::default_random_engine gen;

	// generate random Gaussian noise
	// define normal distributions for sensor noise
	std::normal_distribution<double> Nd_x(0, std_pos[0]);
	std::normal_distribution<double> Nd_y(0, std_pos[1]);
	std::normal_distribution<double> Nd_theta(0, std_pos[2]);

	// Calculate new state
	for(auto& p: particles)
	{
		// add measurements to each particle
		if( fabs(yaw_rate) < 0.0001){  // constant velocity
		  p.x += velocity * delta_t * cos(p.theta);
		  p.y += velocity * delta_t * sin(p.theta);

		} 
		else
		{
		  p.x += velocity / yaw_rate * ( sin( p.theta + yaw_rate*delta_t ) - sin(p.theta) );
		  p.y += velocity / yaw_rate * ( cos( p.theta ) - cos( p.theta + yaw_rate*delta_t ) );
		  p.theta += yaw_rate * delta_t;
		}

		// predicted particles with added sensor noise
		p.x += Nd_x(gen);
		p.y += Nd_y(gen);
		p.theta += Nd_theta(gen);
	}
}

// This function will perform nearest neighbor data association and assign 
// each sensor observation the map Landmark ID associated with it.
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
	/**
	* TODO: Find the predicted measurement that is closest to each 
	*   observed measurement and assign the observed measurement to this 
	*   particular landmark.
	* NOTE: this method will NOT be called by the grading code. But you will 
	*   probably find it useful to implement this method and use it as a helper 
	*   during the updateWeights phase.
	*/
	
	// For each observation 	
	for(auto& obs: observations)
	{
		// init minimum distance to maximum possible
		double minD = std::numeric_limits<float>::max();

		// For each predition 
		for(const auto& pred: predicted)
		{
			// get distance between current/predicted landmarks			
			double distance = dist(obs.x, obs.y, pred.x, pred.y);
			
			// find the predicted landmark nearest the current observed landmark
			if( minD > distance)
			{
				minD = distance;
				obs.id = pred.id;
			}
		}
	}
}

// This function takes the range of the sensor, the landmark measurement uncertainties,
// a vector of landmark measurements, and the map landmarks as input.
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks)
{
	/**
	* TODO: Update the weights of each particle using a mult-variate Gaussian 
	*   distribution. You can read more about this distribution here: 
	*   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	* NOTE: The observations are given in the VEHICLE'S coordinate system. 
	*   Your particles are located according to the MAP'S coordinate system. 
	*   You will need to transform between the two systems. Keep in mind that
	*   this transformation requires both rotation AND translation (but no scaling).
	*   The following is a good resource for the theory:
	*   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	*   and the following is a good resource for the actual equation to implement
	*   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
	*/
   
	// for each particle...
	for(auto& p: particles)
	{
		p.weight = 1.0;

		// step 1: collect valid landmarks
		vector<LandmarkObs> predictions;
		for(const auto& lm: map_landmarks.landmark_list){
		  double distance = dist(p.x, p.y, lm.x_f, lm.y_f);
		  // if the landmark is within the sensor range, save it to predictions
		  if( distance < sensor_range){ 
			predictions.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
		  }
		}

		// step 2: convert observations coordinates from vehicle to map
		vector<LandmarkObs> observations_map;
		double cos_theta = cos(p.theta);
		double sin_theta = sin(p.theta);

		for(const auto& obs: observations)
		{
		  LandmarkObs tmp;
		  tmp.x = obs.x * cos_theta - obs.y * sin_theta + p.x;
		  tmp.y = obs.x * sin_theta + obs.y * cos_theta + p.y;
		  observations_map.push_back(tmp);
		}

		// step 3: find landmark index for each observation
		dataAssociation(predictions, observations_map);

		// step 4: compute the particle's weight:
		for(const auto& obs_m: observations_map){

		  Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs_m.id-1);
		  double x_term = pow(obs_m.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
		  double y_term = pow(obs_m.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
		  double w = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
		  p.weight *=  w;
		}

		weights.push_back(p.weight);
	}
}

// Use the weights of the particles in your particle filter and C++ standard 
// libraries discrete distribution function to update your particles to the Bayesian posterior distribution
void ParticleFilter::resample()
{
	/**
	* TODO: Resample particles with replacement with probability proportional 
	*   to their weight. 
	* NOTE: You may find std::discrete_distribution helpful here.
	*   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	*/
   
	// generate distribution according to weights
	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> dist(weights.begin(), weights.end());

	// create resampled particles
	vector<Particle> resampled_particles;
	resampled_particles.resize(num_particles);

	// resample the particles according to weights
	for(int i=0; i<num_particles; i++){
	int idx = dist(gen);
	resampled_particles[i] = particles[idx];
	}

	// assign the resampled_particles to the previous particles
	particles = resampled_particles;

	// clear the weight vector for the next round
	weights.clear();
	}

/* particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
 * associations: The landmark id that goes along with each listed association
 * sense_x: the associations x mapping already converted to world coordinates
 * sense_y: the associations y mapping already converted to world coordinates
 */
Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}




