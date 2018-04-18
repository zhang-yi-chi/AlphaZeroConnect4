using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class Begin : MonoBehaviour {
    Dropdown dropdownItem;
    InputField ipInput;
    InputField portInput;

    // Load game UI
    public void LoadGameUI(int level)
    {
        dropdownItem = GameObject.Find("Dropdown").GetComponent<Dropdown>();
        ipInput = GameObject.Find("IPInput").GetComponent<InputField>();
        portInput = GameObject.Find("PortInput").GetComponent<InputField>();
        GlobalInfo.IsFirst = dropdownItem.value == 0;
        GlobalInfo.HostIP = ipInput.text;
        GlobalInfo.HostPort = Int32.Parse(portInput.text);
        SceneManager.LoadScene(level);
    }

	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
		
	}
}
