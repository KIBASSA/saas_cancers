import { Component, OnInit, ViewChild, Input } from '@angular/core';
import { ContextMenuComponent, ContextMenuService } from 'ngx-contextmenu';

@Component({
  selector: 'app-context-menu',
  templateUrl: './context-menu.component.html',
  styleUrls: ['./context-menu.component.scss'],
  providers: [ContextMenuService]
})
export class MyContextMenuComponent implements OnInit {

  @ViewChild(ContextMenuComponent, {static: true}) public simpleMenu: ContextMenuComponent;
  @ViewChild(ContextMenuComponent, {static: true}) public actionMenu: ContextMenuComponent;
  
  constructor(private contextMenuService: ContextMenuService) { }

  @Input() contextMenu: ContextMenuComponent;

  ngOnInit() {
  }

  actionMenuClick(action) {
    alert('clicked : ' +  action );
  }

}
