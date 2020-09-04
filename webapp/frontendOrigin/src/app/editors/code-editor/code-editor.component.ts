import { Component, OnInit } from '@angular/core';
import 'codemirror/mode/htmlmixed/htmlmixed.js';
import 'codemirror/mode/shell/shell.js';

@Component({
  selector: 'app-code-editor',
  templateUrl: './code-editor.component.html',
  styleUrls: ['./code-editor.component.scss']
})
export class CodeEditorComponent implements OnInit {

  constructor() { }

  ngOnInit() {
  }

  codeMirrorData1 = `
<!-- Create a simple CodeMirror instance -->
<link rel="stylesheet" href="lib/codemirror.css">
<script src="lib/codemirror.js"></script>
<script>
var editor = CodeMirror.fromTextArea(myTextarea, {
  lineNumbers: true
});
</script>
`;

  codeMirrorData2 = `
#!/bin/bash

# clone the repository
git clone http://github.com/garden/tree

# generate HTTPS credentials
cd tree
openssl genrsa -aes256 -out https.key 1024
`;

}
