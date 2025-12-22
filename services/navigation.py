# services/navigation.py
"""
Navigation service: handles hospital selection and route planning for EVs.
"""
from typing import Dict, Tuple
from Entities.Incident import Incident
from Entities.Hospitals import Hospital
from Entities.ev import EV
from utils.Helpers import utility_navigation


class NavigationService:
    """Service for managing hospital navigation and selection."""
    
    @staticmethod
    def calculate_eta_plus_wait(
        ev: EV,
        hospital: Hospital,
    ) -> float:
        """
        Calculate ETA + total wait time for an EV going to a hospital.
        
        Total wait includes:
        1. Hospital's base wait time (updated each tick)
        2. Service time of higher priority EVs already being served
        3. Service time of same priority EVs already being served
        
        Args:
            ev: The EV (for location and priority)
            hospital: The target hospital
            
        Returns:
            ETA + total wait time in minutes
        """
        # Calculate ETA from EV to hospital
        ev_lat, ev_lng = ev.location
        eta = hospital.estimate_eta_minutes(ev_lat, ev_lng, kmph=40.0)
        
        # If EV has already been assigned to this hospital and reached it, return stored wait time
        if ev.navTargetHospitalId == hospital.id and ev.navWaitTime >= 0:
            return ev.navWaitTime
        
        # Otherwise calculate full wait time (for hospital selection)
        base_wait = float(getattr(hospital, "waitTime", 0.0))
        ev_priority = getattr(ev, 'assignedPatientPriority', 2)
        
        # Add service time of higher priority EVs
        total_additional_wait = 0.0
        num_evs1 = len(getattr(hospital, 'evs_serving_priority_1', []))
        #print("number of p1 evs waiting at HC" ,hospital.id,"queue",num_evs1)
        num_evs2 = len(getattr(hospital, 'evs_serving_priority_2', []))
        #print("number of p2 evs waiting at HC" ,hospital.id,"queue",num_evs2)
        num_evs3 = len(getattr(hospital, 'evs_serving_priority_3', []))
        #print("number of p3 evs waiting at HC" ,hospital.id,"queue",num_evs3)
        for priority in range(1, ev_priority):
            if priority == 1:
                total_additional_wait = num_evs1 * 8.0  # Assume 8 minutes per EV service
                

            elif priority == 2:
                total_additional_wait = (num_evs1 + num_evs2) * 8.0
            else: 
                total_additional_wait = (num_evs1 + num_evs2 + num_evs3) * 8.0  # Assume 8 minutes per EV service
        
        # Add service time of same priority EVs already being served
        '''if ev_priority == 1:
            same_priority_evs = len(getattr(hospital, 'evs_serving_priority_1', []))
        elif ev_priority == 2:
            same_priority_evs = len(getattr(hospital, 'evs_serving_priority_2', []))
        elif ev_priority == 3:
            same_priority_evs = len(getattr(hospital, 'evs_serving_priority_3', []))
        else:
            same_priority_evs = 0
        
        total_additional_wait += same_priority_evs * 8.0'''
        
        # Total wait = base wait + service times of other EVs
        total_wait = base_wait + total_additional_wait
        #print("wait time at",hospital.id,total_additional_wait,"priority",ev_priority)
        #print("service times at hc",hospital.id,base_wait)
        
        return eta + total_wait
    
    @staticmethod
    def select_hospital(
        ev: EV,
        #incident: Incident,
        hospitals_in_grid: Hospital,
        calculate_wait_func
    ) -> Hospital:
        """
        Select the best hospital from a list based on minimum wait time.
        
        Args:
            ev: The EV that will go to hospital
            incident: The incident (patient) information
            hospitals_in_grid: List of hospitals in the destination grid
            calculate_wait_func: Function to calculate eta + wait (from MAP)
            
        Returns:
            Best hospital (Hospital object) or None if no hospitals available
        """

        
        # Pick hospital with minimum total wait (eta + queue)
        best_hospital = min(
            hospitals_in_grid,
            key=lambda h: calculate_wait_func(ev, h)
        )
        
        # Get patient priority
        priority = getattr(ev.assignedPatientPriority, 'priority', 1)
        
        # Add EV to hospital's priority-specific service list
        best_hospital.start_service(ev_id=ev.id, priority=priority)
        #print("best hospital",best_hospital)
        
        # Set nav wait time to the calculated wait for this hospital
        ev.navWaitTime = calculate_wait_func(ev, best_hospital)
        
        return best_hospital


    
    @staticmethod
    def get_candidate_hospitals(
        incident: Incident,
        hospitals: Dict[int, Hospital],
        max_k: int = 8,
    ) -> Tuple[list[int], list[float], list[float]]:
        """
        Get the top K nearest hospitals for a patient incident.
        
        Useful for decision-making systems that need multiple options
        (e.g., reinforcement learning agents).
        
        Args:
            incident: The incident/patient location
            hospitals: Dict mapping hospital IDs to Hospital objects
            max_k: Maximum number of hospitals to return
            
        Returns:
            Tuple of (hospital_ids, etas_minutes, wait_times)
        """
        patient_lat, patient_lng = incident.location
        return Hospital.select_nearest_hospitals(
            hospitals,
            patient_lat,
            patient_lng,
            max_k=max_k,
        )
