import { Component, OnInit } from '@angular/core';
import { ToastrService } from 'ngx-toastr';

@Component({
  selector: 'app-notifications',
  templateUrl: './notifications.component.html',
  styleUrls: ['./notifications.component.scss']
})
export class NotificationsComponent implements OnInit {

  constructor(private toastr: ToastrService) { 
  }

 ngOnInit() {
 }

 showSuccessToast() {
   this.toastr.success("And these were just the basic demos! Scroll down to check further details on how to customize the output", 'Success', {
     progressBar: true,
     closeButton: true
   })
 }

 showInfoToast() {
   this.toastr.info("And these were just the basic demos! Scroll down to check further details on how to customize the output", 'Info', {
     progressBar: true,
     closeButton: true
   })
 }

 showWarningToast() {
   this.toastr.warning("And these were just the basic demos! Scroll down to check further details on how to customize the output", 'Warning', {
     progressBar: true,
     closeButton: true
   })
 }

 showDangerToast() {
   this.toastr.error("And these were just the basic demos! Scroll down to check further details on how to customize the output", 'Error', {
     progressBar: true,
     closeButton: true
   })
 }

 showToastPosition (position) {
   switch(position) {
     case 'bottom-left':
       this.toastr.info("Specify the custom position object or use one of the predefined ones", "Positioning", {
         progressBar: true,
         closeButton: true,
         positionClass: 'toast-bottom-left'
       });
       break;
     case 'bottom-right':
       this.toastr.info("Specify the custom position object or use one of the predefined ones", "Positioning", {
         progressBar: true,
         closeButton: true,
         positionClass: 'toast-bottom-right'
       });
       break;
     case 'bottom-center':
       this.toastr.info("Specify the custom position object or use one of the predefined ones", "Positioning", {
         progressBar: true,
         closeButton: true,
         positionClass: 'toast-bottom-center'
       });
       break;
     case 'top-left':
       this.toastr.info("Specify the custom position object or use one of the predefined ones", "Positioning", {
         progressBar: true,
         closeButton: true,
         positionClass: 'toast-top-left'
       });
       break;
     case 'top-right':
       this.toastr.info("Specify the custom position object or use one of the predefined ones", "Positioning", {
         progressBar: true,
         closeButton: true,
         positionClass: 'toast-top-right'
       });
       break;
     case 'top-center':
       this.toastr.info("Specify the custom position object or use one of the predefined ones", "Positioning", {
         progressBar: true,
         closeButton: true,
         positionClass: 'toast-top-center'
       });
       break;case 'bottom-full-width':
       this.toastr.info("Specify the custom position object or use one of the predefined ones", "Positioning", {
         progressBar: true,
         closeButton: true,
         positionClass: 'toast-bottom-full-width'
       });
       break;case 'top-full-width':
       this.toastr.info("Specify the custom position object or use one of the predefined ones", "Positioning", {
         progressBar: true,
         closeButton: true,
         positionClass: 'toast-top-full-width'
       });
       break;
   }
 }

}
