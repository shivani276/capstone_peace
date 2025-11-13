# services/navigation.py
"""
Navigation service: handles hospital selection and route planning for EVs.
"""
from typing import Dict, Tuple
from Entities.Incident import Incident
from Entities.Hospitals import Hospital


class NavigationService:
    """Service for managing hospital navigation and selection."""
    
    @staticmethod
    def select_hospital_for_incident(
        incident: Incident,
        hospitals: Dict[int, Hospital],
    ) -> Tuple[int, float]:

        best_hid, best_eta = -1, float("inf")
        patient_lat, patient_lng = incident.location
        
        for hid, hc in hospitals.items():
            eta = hc.estimate_eta_minutes(patient_lat, patient_lng, kmph=40.0)
            if eta < best_eta:
                best_eta, best_hid = eta, hid
        
        return best_hid, best_eta
    
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
